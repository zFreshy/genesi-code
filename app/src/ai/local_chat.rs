//! Login-free local AI chat.
//!
//! Talks DIRECTLY to a local OpenAI-compatible endpoint (ollama / llama-server —
//! the same one `genesi-ai-mode` runs at `http://localhost:11434`). This
//! deliberately bypasses the Warp cloud agent runtime, which is server-side and
//! event-sourced (every turn is a `POST /agent/runs` against the Warp servers
//! and requires an account). Pointing that path at `localhost` cannot work
//! because the base URL is resolved on Warp's servers, not the user's machine.
//! So for Genesi we own this small client (and the chat UI on top of it), giving
//! AI with no account and no cloud.
#![allow(dead_code)]

use std::path::PathBuf;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures::stream::{Stream, StreamExt};
use reqwest_eventsource::{Error as EventSourceError, Event};
use serde::{Deserialize, Serialize};

// NOTE: there is deliberately no whole-request timeout on the chat paths. An
// earlier 120s cap aborted slow-but-working generations on CPU (the "error
// sending request" reports), so both streams now run until the server finishes
// or the user presses Stop. Connection failures still surface immediately.

/// Default local endpoint: ollama's OpenAI-compatible API.
pub const DEFAULT_LOCAL_BASE_URL: &str = "http://localhost:11434/v1";

/// A single chat message in the OpenAI `chat/completions` shape.
///
/// Serialization is hand-written because the wire shape depends on the payload:
/// a text-only message sends `content` as a plain string (which every
/// OpenAI-compatible server accepts, including old llama-server builds), and
/// only a message carrying an image switches to the content-parts array.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    /// Images attached to this message. Empty for every ordinary message; only
    /// the turn the user attached a picture to carries any.
    #[serde(default, skip)]
    pub images: Vec<ChatImage>,
}

/// An image being sent to a vision-capable model, already base64-encoded.
#[derive(Debug, Clone)]
pub struct ChatImage {
    pub mime_type: String,
    pub base64: String,
}

impl ChatImage {
    /// The `data:` URI form both OpenAI-compatible servers and llama-server take.
    pub fn data_uri(&self) -> String {
        format!("data:{};base64,{}", self.mime_type, self.base64)
    }
}

impl Serialize for ChatMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("role", &self.role)?;
        if self.images.is_empty() {
            map.serialize_entry("content", &self.content)?;
        } else {
            let mut parts = vec![serde_json::json!({
                "type": "text",
                "text": self.content,
            })];
            parts.extend(self.images.iter().map(|image| {
                serde_json::json!({
                    "type": "image_url",
                    "image_url": { "url": image.data_uri() },
                })
            }));
            map.serialize_entry("content", &parts)?;
        }
        map.end()
    }
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".to_string(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".to_string(),
            content: content.into(),
            images: Vec::new(),
        }
    }

    pub fn with_images(mut self, images: Vec<ChatImage>) -> Self {
        self.images = images;
        self
    }
}

/// Caps on how much of the active file is injected as context. A small local
/// model has a tight context window, so the whole-file case is bounded; a
/// selection is sent verbatim (the user picked it deliberately).
const CONTEXT_MAX_LINES: usize = 200;
const CONTEXT_MAX_CHARS: usize = 6000;

/// Snapshot of the code editor the user is focused on, attached to a chat turn
/// so the local model can answer about the code in front of the user — like a
/// normal AI IDE. The workspace builds this at send time (see
/// `Workspace::focused_code_context`) so it's always fresh.
#[derive(Debug, Clone)]
pub struct CodeContext {
    /// Display name of the file (the file name, not the full path).
    pub path: String,
    /// Language hint for the markdown code fence (from the file extension).
    pub language: String,
    /// A user selection: `(start_line, end_line, text)`, 1-indexed lines.
    pub selection: Option<(u32, u32, String)>,
    /// The full file text, used only when there's no selection.
    pub file_text: Option<String>,
}

impl CodeContext {
    /// Short label for the chat UI, e.g. `foo.rs · lines 12-40`.
    pub fn label(&self) -> String {
        match &self.selection {
            Some((start, end, _)) if start == end => format!("{} · line {start}", self.path),
            Some((start, end, _)) => format!("{} · lines {start}-{end}", self.path),
            None => self.path.clone(),
        }
    }

    /// Render the context as a system message prepended to the turn. Bounded so a
    /// large file can't blow a small local model's context window.
    pub fn to_system_message(&self) -> ChatMessage {
        let mut body = String::from(
            "The user is editing code in Genesi Code. Use the file context below to \
             answer their question; refer to it only as needed.\n\n",
        );
        match &self.selection {
            Some((start, end, text)) => {
                body.push_str(&format!(
                    "File: {} (selected lines {start}-{end})\n",
                    self.path
                ));
                body.push_str(&fenced(&self.language, &clamp_context(text)));
            }
            None => {
                body.push_str(&format!("File: {}\n", self.path));
                let text = self.file_text.as_deref().unwrap_or_default();
                body.push_str(&fenced(&self.language, &clamp_context(text)));
            }
        }
        ChatMessage::system(body)
    }
}

/// An attachment is bigger than the incidental editor context: the user picked
/// this file deliberately, so it gets a larger slice of the window.
const ATTACHMENT_MAX_LINES: usize = 600;
const ATTACHMENT_MAX_CHARS: usize = 24_000;

/// Extensions treated as images (offered to the model only when it can see).
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp"];

/// What an attachment is, which decides how it reaches the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttachmentKind {
    /// Read as text and fenced into the prompt.
    Text,
    /// Base64'd into an image content part, for a model that can see.
    Image,
}

/// A file the user attached to the conversation by hand — via the paperclip, or
/// by pasting a file copied in the file manager.
///
/// Deliberately separate from [`CodeContext`]: that one is the editor's current
/// file, refreshed and attached implicitly on every turn, while these are pinned
/// references the user chose and can remove.
#[derive(Debug, Clone)]
pub struct ChatAttachment {
    /// Absolute path on disk. `None` for an image pasted as raw clipboard bytes,
    /// which has no file behind it.
    pub path: Option<PathBuf>,
    /// What the chip shows — the file name, or a generated one for a paste.
    pub name: String,
    pub kind: AttachmentKind,
    /// Raw bytes for a pasted image. Files are read at send time instead, so a
    /// long-lived attachment always reflects what is on disk now.
    pub bytes: Option<Vec<u8>>,
}

impl ChatAttachment {
    /// Classify and wrap a path the user picked or pasted.
    pub fn from_path(path: PathBuf) -> Self {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        let kind = if is_image_path(&path) {
            AttachmentKind::Image
        } else {
            AttachmentKind::Text
        };
        Self {
            path: Some(path),
            name,
            kind,
            bytes: None,
        }
    }

    /// An image pasted straight from the clipboard (a screenshot), which exists
    /// only as bytes.
    pub fn from_image_bytes(name: String, mime_type: &str, bytes: Vec<u8>) -> Self {
        let name = if name.trim().is_empty() {
            let ext = mime_type.rsplit('/').next().unwrap_or("png");
            format!("pasted-image.{ext}")
        } else {
            name
        };
        Self {
            path: None,
            name,
            kind: AttachmentKind::Image,
            bytes: Some(bytes),
        }
    }

    /// Language hint for the code fence, from the extension.
    fn language(&self) -> String {
        self.path
            .as_ref()
            .and_then(|p| p.extension())
            .map(|ext| ext.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// Read the attachment's bytes: the pasted buffer, or the file as it is on
    /// disk right now.
    pub fn read_bytes(&self) -> Option<Vec<u8>> {
        if let Some(bytes) = &self.bytes {
            return Some(bytes.clone());
        }
        std::fs::read(self.path.as_ref()?).ok()
    }

    /// The image encoded for a vision request, or `None` if it isn't an image or
    /// can't be read.
    pub fn as_chat_image(&self) -> Option<ChatImage> {
        if self.kind != AttachmentKind::Image {
            return None;
        }
        let bytes = self.read_bytes()?;
        use base64::Engine as _;
        Some(ChatImage {
            mime_type: mime_type_for(&self.name).to_string(),
            base64: base64::engine::general_purpose::STANDARD.encode(bytes),
        })
    }

    /// Render a TEXT attachment as a system message. Returns `None` for an image
    /// (which travels as an image part) or a file that can't be read as text.
    pub fn to_system_message(&self) -> Option<ChatMessage> {
        if self.kind != AttachmentKind::Text {
            return None;
        }
        let path = self.path.as_ref()?;
        let raw = std::fs::read(path).ok()?;
        // A binary file fenced into the prompt is noise that costs context and
        // teaches the model nothing — say what it is instead.
        let Ok(text) = String::from_utf8(raw) else {
            return Some(ChatMessage::system(format!(
                "The user attached `{}`, which is a binary file ({}). Its contents \
                 are not included.",
                self.name,
                path.display()
            )));
        };
        let mut body = format!(
            "The user attached this file for reference. Full path: {}\n\nFile: {}\n",
            path.display(),
            self.name
        );
        body.push_str(&fenced(
            &self.language(),
            &clamp(&text, ATTACHMENT_MAX_LINES, ATTACHMENT_MAX_CHARS),
        ));
        Some(ChatMessage::system(body))
    }
}

pub fn is_image_path(path: &std::path::Path) -> bool {
    path.extension()
        .map(|ext| ext.to_string_lossy().to_ascii_lowercase())
        .is_some_and(|ext| IMAGE_EXTENSIONS.contains(&ext.as_str()))
}

fn mime_type_for(name: &str) -> &'static str {
    match name
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        _ => "image/png",
    }
}

/// Whether `model` is one that can actually look at an image.
///
/// A name check, because no local endpoint reliably reports modality: ollama's
/// `/v1/models` gives ids only, and llama-server can't say whether an mmproj is
/// loaded. Sending an image to a text-only model wastes the context window and
/// makes some servers reject the whole request, so the default when a name isn't
/// recognised is NO — the attachment is still named in the prompt, just not
/// pixel-encoded.
pub fn model_supports_vision(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    const VISION_MARKERS: &[&str] = &[
        // Local vision stacks.
        "llava",
        "bakllava",
        "moondream",
        "minicpm-v",
        "llama3.2-vision",
        "llama-3.2-vision",
        "pixtral",
        "internvl",
        "cogvlm",
        "glm-4v",
        "-vl",
        "_vl",
        "vl-",
        "vision",
        "gemma-3",
        "gemma3",
        "mistral-small-3",
        "granite-vision",
        // Cloud families that are multimodal across the board.
        "gpt-4o",
        "gpt-4.1",
        "gpt-5",
        "o3",
        "o4-mini",
        "claude",
        "gemini",
    ];
    VISION_MARKERS.iter().any(|marker| model.contains(marker))
}

/// Truncate context text to the line/char caps, marking it when cut.
fn clamp_context(text: &str) -> String {
    clamp(text, CONTEXT_MAX_LINES, CONTEXT_MAX_CHARS)
}

/// Truncate text to a line and character cap, marking it when cut.
fn clamp(text: &str, max_lines: usize, max_chars: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let line_truncated = lines.len() > max_lines;
    let line_capped = lines[..lines.len().min(max_lines)].join("\n");

    let (mut out, char_truncated) = if line_capped.chars().count() > max_chars {
        (
            line_capped.chars().take(max_chars).collect::<String>(),
            true,
        )
    } else {
        (line_capped, false)
    };
    if line_truncated || char_truncated {
        out.push_str("\n… [truncated]");
    }
    out
}

/// Wrap text in a markdown code fence with a language hint.
fn fenced(language: &str, text: &str) -> String {
    format!("```{language}\n{text}\n```\n")
}

/// One streamed update from the model.
#[derive(Debug, Clone)]
pub enum ChatStreamItem {
    /// A chunk of assistant text to append.
    Token(String),
    /// A chunk of a reasoning model's chain-of-thought. Kept SEPARATE from
    /// `Token` because the agent loop parses tool calls out of the answer: mixing
    /// the model's thinking into that buffer means it parses narration ("I will
    /// create the file…") instead of the `<tool:…>` tag it was looking for.
    Reasoning(String),
    /// A fragment of a NATIVE (OpenAI-shaped) tool call. A harmony model like
    /// gpt-oss writes the act of calling a tool to its `commentary` channel, and
    /// llama-server's parser turns that into `delta.tool_calls` — never into
    /// `content`. Ignoring this field is why "answer me" worked and "do
    /// something" produced an empty reply: the model DID act, the call just went
    /// down a channel nothing was reading.
    ///
    /// Fragments arrive split across chunks (the name once, the JSON arguments a
    /// few characters at a time), so the caller accumulates them and builds the
    /// call when the step ends.
    ToolCall {
        name: Option<String>,
        arguments: String,
    },
    /// The model finished this turn cleanly.
    Done,
}

#[derive(Serialize)]
struct ChatRequest<'a> {
    /// Omitted for the local Turbo server: llama-server has exactly ONE model
    /// open and newer builds reject a `model` that doesn't match the alias they
    /// advertise. The AI Mode Monitor has always omitted it on this path, which
    /// is why its Turbo chat worked while Code's silently produced nothing.
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    messages: &'a [ChatMessage],
    stream: bool,
    /// Without a cap llama-server can run to the end of the context window; the
    /// Monitor sends one and Code did not.
    max_tokens: u32,
    /// Reuse the KV prefix across turns — the system prompt and file context are
    /// resent every turn, so this is a large time-to-first-token win locally.
    /// Ignored by cloud providers that don't know the field.
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    cache_prompt: bool,
    /// Function schemas for a model that calls tools natively. Omitted entirely
    /// in plain chat, and dropped for good once a server rejects it (llama-server
    /// needs `--jinja` to accept this field at all).
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<serde_json::Value>,
}

/// Whether a server error is the "I don't do tools" refusal, meaning the request
/// should be retried without the `tools` field rather than shown to the user.
/// llama-server answers a tools request built without `--jinja` with exactly
/// that complaint.
pub fn is_tools_unsupported_error(message: &str) -> bool {
    let lowered = message.to_ascii_lowercase();
    (lowered.contains("tool") || lowered.contains("function"))
        && (lowered.contains("jinja")
            || lowered.contains("not supported")
            || lowered.contains("unsupported")
            || lowered.contains("unknown"))
}

/// Upper bound on one local reply. Generous enough for a full code block, small
/// enough that a runaway model can't fill the whole context window.
///
/// A reasoning model spends this budget TWICE over: once on its private analysis
/// and once on the answer. At 2048 a model like gpt-oss-20b could think its way
/// through a task, hit the cap mid-thought, and stop before writing a single
/// token of `content` — which the agent loop then reported as "the model
/// returned nothing".
///
/// This is the CEILING, not the value sent: the caller reduces it to what the
/// server's context window can actually spare. Asking for 4096 reply tokens on a
/// 4096-token window leaves no room for the prompt at all.
pub const LOCAL_MAX_TOKENS: u32 = 4096;

/// The reply budget on a server whose window is `n_ctx`, given a prompt we
/// estimate at `prompt_tokens`. Never returns zero — a tiny reply beats an error.
pub fn reply_budget(n_ctx: u32, prompt_tokens: usize) -> u32 {
    let free = (n_ctx as usize).saturating_sub(prompt_tokens);
    (free as u32).min(LOCAL_MAX_TOKENS).max(256)
}

#[derive(Serialize)]
struct AnthropicChatRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: &'a [AnthropicInputMessage],
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

#[derive(Debug, Clone)]
struct AnthropicInputMessage {
    role: String,
    content: String,
    images: Vec<ChatImage>,
}

/// Anthropic takes its own block shape (`source.data`, not a `data:` URI), so
/// this mirrors the OpenAI serializer rather than sharing it.
impl Serialize for AnthropicInputMessage {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;

        let mut map = serializer.serialize_map(Some(2))?;
        map.serialize_entry("role", &self.role)?;
        if self.images.is_empty() {
            map.serialize_entry("content", &self.content)?;
        } else {
            let mut blocks: Vec<serde_json::Value> = self
                .images
                .iter()
                .map(|image| {
                    serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image.mime_type,
                            "data": image.base64,
                        },
                    })
                })
                .collect();
            blocks.push(serde_json::json!({ "type": "text", "text": self.content }));
            map.serialize_entry("content", &blocks)?;
        }
        map.end()
    }
}

#[derive(Deserialize)]
struct ChatChunk {
    #[serde(default)]
    choices: Vec<ChatChoice>,
}

#[derive(Deserialize)]
struct ChatChoice {
    #[serde(default)]
    delta: ChatDelta,
}

#[derive(Deserialize, Default)]
struct ChatDelta {
    #[serde(default)]
    content: Option<String>,
    /// Reasoning models (gpt-oss's harmony channels, DeepSeek-R1, Qwen3 in
    /// thinking mode) have llama-server put their chain-of-thought here and
    /// leave `content` empty until the final answer. Reading only `content`
    /// meant such a model streamed for a while and appeared to answer nothing.
    #[serde(default)]
    reasoning_content: Option<String>,
    /// llama.cpp also accepts `reasoning` for the same field on some builds.
    #[serde(default)]
    reasoning: Option<String>,
    /// Native tool calls — see [`ChatStreamItem::ToolCall`].
    #[serde(default)]
    tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Deserialize, Default)]
struct ToolCallDelta {
    #[serde(default)]
    function: Option<FunctionCallDelta>,
}

#[derive(Deserialize, Default)]
struct FunctionCallDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

/// One message in ollama's NATIVE shape. Images are a flat array of bare base64
/// strings on the message itself — not OpenAI content parts — so the OpenAI
/// serializer on [`ChatMessage`] can't be reused here.
#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    images: Vec<String>,
}

impl From<ChatMessage> for OllamaMessage {
    fn from(message: ChatMessage) -> Self {
        Self {
            role: message.role,
            content: message.content,
            images: message
                .images
                .into_iter()
                .map(|image| image.base64)
                .collect(),
        }
    }
}

/// Native (non-OpenAI) request body for ollama's `/api/chat`.
#[derive(Serialize)]
struct OllamaChatRequest<'a> {
    model: &'a str,
    messages: &'a [OllamaMessage],
    stream: bool,
    /// Keep the model resident this long after the reply, so the next message
    /// (and the AI Mode Monitor, which shares the same ollama model) doesn't pay
    /// a multi-second cold reload. ollama's default is only 5m, which expires
    /// during normal back-and-forth and makes Code feel far slower than it is.
    keep_alive: &'static str,
}

/// How long ollama should keep the model loaded between requests.
const OLLAMA_KEEP_ALIVE: &str = "30m";

/// One NDJSON line of ollama's native `/api/chat` stream: a `message.content`
/// token and, on the last line, `done: true`.
#[derive(Deserialize, Default)]
struct OllamaChatResponse {
    #[serde(default)]
    message: OllamaResponseMessage,
    /// True on the final streamed line.
    #[serde(default)]
    done: bool,
    /// Set when ollama itself errors (e.g. model not found).
    #[serde(default)]
    error: Option<String>,
}

#[derive(Deserialize, Default)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: String,
}

/// ollama's native chat endpoint. Its OpenAI-compat `/v1/chat/completions`
/// hangs on some versions (a raw curl to it never responds, while the native
/// API and `ollama run` work) — the AI Mode monitor talks to the native API for
/// exactly this reason, so we mirror it.
const OLLAMA_NATIVE_CHAT_URL: &str = "http://localhost:11434/api/chat";

/// Stream a chat completion from the given local endpoint. Yields assistant text
/// chunks (`Token`) and a terminal `Done`. No auth headers are sent.
///
/// The two endpoints speak different protocols: ollama uses its native
/// `/api/chat` (its OpenAI-compat layer is unreliable), while genesi-ai-turbo's
/// llama-server uses the OpenAI-compatible `/v1/chat/completions` SSE stream.
pub fn stream_chat(
    endpoint: LocalEndpoint,
    model: &str,
    messages: Vec<ChatMessage>,
    tools: Option<serde_json::Value>,
    reply_tokens: u32,
) -> futures::stream::BoxStream<'static, Result<ChatStreamItem>> {
    match endpoint {
        // ollama's native API takes tools in its own shape and its support varies
        // per model; the text protocol is what works there, so leave it alone.
        LocalEndpoint::Ollama => stream_chat_ollama_native(model.to_string(), messages).boxed(),
        LocalEndpoint::Turbo => stream_chat_openai_sse(
            endpoint.base_url(),
            None,
            messages,
            None,
            tools,
            reply_tokens,
        )
        .boxed(),
    }
}

/// Stream a chat completion from a user-configured cloud provider (BYOK). Any
/// OpenAI-compatible endpoint works — OpenAI, OpenRouter, Groq, Together, or
/// Hugging Face's free router (`https://router.huggingface.co/v1`) — with the
/// user's own API key sent as a bearer token. Local stays the default; this is
/// purely opt-in for a bigger/faster model than the machine can run.
pub fn stream_chat_cloud(
    provider: CloudProviderKind,
    model: &str,
    api_key: String,
    messages: Vec<ChatMessage>,
    tools: Option<serde_json::Value>,
    reply_tokens: u32,
) -> futures::stream::BoxStream<'static, Result<ChatStreamItem>> {
    let model = model.to_string();
    match provider {
        CloudProviderKind::Anthropic => retrying_on_rate_limit(Box::new(move || {
            stream_chat_anthropic_sse(&model, messages.clone(), api_key.clone()).boxed()
        })),
        _ => {
            let base_url = provider.base_url();
            retrying_on_rate_limit(Box::new(move || {
                stream_chat_openai_sse(
                    base_url,
                    Some(&model),
                    messages.clone(),
                    Some(api_key.clone()),
                    tools.clone(),
                    reply_tokens,
                )
                .boxed()
            }))
        }
    }
}

/// How many times a refused turn is retried before the error reaches the user.
/// Two rides out a per-minute window, or a provider's brief capacity dip,
/// without leaving anyone staring at a panel that looks frozen.
const RATE_LIMIT_MAX_RETRIES: u32 = 2;

/// Longest pause we'll honour before deciding the user would rather see the
/// error than keep waiting.
const RATE_LIMIT_MAX_WAIT: Duration = Duration::from_secs(45);

/// What to wait when a provider says "slow down" without saying for how long.
const RATE_LIMIT_FALLBACK_WAIT: Duration = Duration::from_secs(10);

/// What to wait after a transient server error. Shorter than the rate-limit
/// fallback: nothing is metered here, the far side is just briefly unwell, and
/// doubling each attempt covers a longer dip without a long first stall.
const SERVER_ERROR_FIRST_WAIT: Duration = Duration::from_secs(2);

/// Retry a chat stream the provider refused for a reason that will pass.
///
/// Two things refuse a turn that would have worked a moment later. Every BYOK
/// provider meters requests, and the free tiers meter them tightly: Groq's is
/// 8k tokens per MINUTE, which one agent step can spend on its own. And the
/// same free tiers shed load under bursts — a 503 while a model is cold, or a
/// 502 from a gateway mid-rollout. An agent turn issues its steps back to back,
/// which is exactly the shape that trips both.
///
/// Only the rate limit was handled before, so a 503 killed the turn outright:
/// the panel streamed a few seconds of a first step and then went red, which is
/// what "it works for a few seconds and then errors" was.
///
/// `build` is called once per attempt, so it must own everything it needs. It
/// hands back an already-boxed stream on purpose: making this generic over the
/// stream type monomorphised one `async_stream` generator per provider path,
/// each wrapping the last, and rustc blew its stack compiling the result.
///
/// Only a refusal that lands BEFORE any output is retried: once tokens have
/// been streamed into the UI, replaying the request would duplicate them.
fn retrying_on_rate_limit(
    build: Box<dyn Fn() -> futures::stream::BoxStream<'static, Result<ChatStreamItem>> + Send>,
) -> futures::stream::BoxStream<'static, Result<ChatStreamItem>> {
    async_stream::stream! {
        for attempt in 0..=RATE_LIMIT_MAX_RETRIES {
            let mut inner = build();
            let mut streamed_anything = false;
            let mut retry_after = None;
            while let Some(item) = inner.next().await {
                if !streamed_anything && attempt < RATE_LIMIT_MAX_RETRIES {
                    if let Some(wait) = item.as_ref().err().and_then(|err| {
                        retry_wait(&err.to_string(), attempt)
                    }) {
                        retry_after = Some(wait);
                        break;
                    }
                }
                streamed_anything = true;
                yield item;
            }
            let Some(wait) = retry_after else { return };
            log::info!(
                "local chat: provider refused the request, retrying in {:.1}s",
                wait.as_secs_f64()
            );
            warpui::r#async::Timer::after(wait).await;
        }
    }
    .boxed()
}

/// How long to wait before retrying, or `None` when the error will not pass.
///
/// Returning `None` is what stops a bad key, a missing model, or a prompt that
/// overflows the window from being sent three times.
fn retry_wait(error: &str, attempt: u32) -> Option<Duration> {
    let error = error.to_ascii_lowercase();

    // A rate limit states its own wait ("Please try again in 22.627499999s"),
    // and honouring it is the difference between the retry working and it
    // being refused again.
    if error.contains("429") || error.contains("too many requests") || error.contains("rate limit")
    {
        let wait = error
            .split("try again in")
            .nth(1)
            .and_then(parse_leading_duration)
            .unwrap_or(RATE_LIMIT_FALLBACK_WAIT);
        return Some(wait.min(RATE_LIMIT_MAX_WAIT));
    }

    // Transient server-side failures. 500 is deliberately absent: providers
    // return it for malformed requests too, and retrying one of those just
    // spends the user's quota three times to reach the same error.
    let transient = ["502", "503", "504"]
        .iter()
        .any(|code| error.contains(code))
        || error.contains("service unavailable")
        || error.contains("bad gateway")
        || error.contains("gateway timeout")
        || error.contains("overloaded")
        || error.contains("temporarily unavailable")
        || error.contains("currently loading");
    if !transient {
        return None;
    }

    let wait = SERVER_ERROR_FIRST_WAIT * 2u32.pow(attempt);
    Some(wait.min(RATE_LIMIT_MAX_WAIT))
}

/// Read a leading "22.6s" / "500ms" / "1m" off the front of `text`.
fn parse_leading_duration(text: &str) -> Option<Duration> {
    let text = text.trim_start();
    // Digits and '.' are single-byte, so the char count is also the byte offset.
    let number: String = text
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let value: f64 = number.parse().ok()?;
    let unit = text[number.len()..].trim_start();
    // "ms" before "m": otherwise a 500ms wait is read as 500 minutes.
    let seconds = if unit.starts_with("ms") {
        value / 1000.0
    } else if unit.starts_with('m') {
        value * 60.0
    } else {
        value
    };
    (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds))
}

fn cloud_http_client() -> http_client::Client {
    http_client::Client::from_client_builder(reqwest::Client::builder().no_proxy())
        .unwrap_or_else(|_| http_client::Client::new())
}

/// ollama's native `/api/chat` with `stream: true`: parse the NDJSON token
/// stream and yield tokens as they arrive. The OpenAI-compat endpoint hangs on
/// some ollama versions, so we use the native API the AI Mode monitor uses.
///
/// NOTE: deliberately NO total `.timeout()`. A GPU-less 7B streams for a while,
/// and ollama emits tokens continuously once connected, so a whole-response
/// deadline (the old 120s) aborted slow-but-working generations — exactly the
/// "error sending request" failure. Streaming removes that wall; an unreachable
/// ollama still errors quickly on `send()`.
fn stream_chat_ollama_native(
    model: String,
    messages: Vec<ChatMessage>,
) -> impl Stream<Item = Result<ChatStreamItem>> {
    async_stream::stream! {
        let messages: Vec<OllamaMessage> = messages.into_iter().map(Into::into).collect();
        let body = OllamaChatRequest {
            model: &model,
            messages: &messages,
            stream: true,
            keep_alive: OLLAMA_KEEP_ALIVE,
        };
        let client = http_client::Client::new();
        let sent = client.post(OLLAMA_NATIVE_CHAT_URL).json(&body).send().await;
        let response = match sent {
            Ok(response) => response,
            Err(e) => {
                yield Err(anyhow!(
                    "failed to reach ollama at {OLLAMA_NATIVE_CHAT_URL}: {}",
                    error_chain(&e)
                ));
                return;
            }
        };
        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            yield Err(anyhow!("ollama returned {status}: {detail}"));
            return;
        }

        // ollama streams newline-delimited JSON, one object per token.
        let mut stream = response.bytes_stream();
        let mut buffer: Vec<u8> = Vec::new();
        while let Some(chunk) = stream.next().await {
            let chunk = match chunk {
                Ok(chunk) => chunk,
                Err(e) => {
                    yield Err(anyhow!("ollama stream error: {}", error_chain(&e)));
                    return;
                }
            };
            buffer.extend_from_slice(&chunk);
            while let Some(newline) = buffer.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buffer.drain(..=newline).collect();
                let line = &line[..line.len() - 1]; // drop the newline
                if line.iter().all(u8::is_ascii_whitespace) {
                    continue;
                }
                match serde_json::from_slice::<OllamaChatResponse>(line) {
                    Ok(parsed) => {
                        if let Some(err) = parsed.error {
                            yield Err(anyhow!("ollama error: {err}"));
                            return;
                        }
                        if !parsed.message.content.is_empty() {
                            yield Ok(ChatStreamItem::Token(parsed.message.content));
                        }
                        if parsed.done {
                            yield Ok(ChatStreamItem::Done);
                            return;
                        }
                    }
                    Err(e) => log::warn!("ollama: skipping malformed NDJSON line: {e}"),
                }
            }
        }
        yield Ok(ChatStreamItem::Done);
    }
}

/// Walk an error's source chain into one string so callers see the real cause
/// (e.g. "connection refused") instead of reqwest's opaque "error sending
/// request for url".
fn error_chain<E: std::error::Error>(err: &E) -> String {
    let mut out = err.to_string();
    let mut source = err.source();
    while let Some(cause) = source {
        out.push_str(&format!(": {cause}"));
        source = cause.source();
    }
    out
}

/// OpenAI-compatible SSE streaming (`/v1/chat/completions`) — used for the Turbo
/// (llama-server) endpoint, whose OpenAI compat layer works correctly.
/// `model` is `None` for the local llama-server (it has exactly one model open
/// and newer builds reject a mismatched name) and `Some(..)` for a cloud
/// provider, which requires it.
fn stream_chat_openai_sse(
    base_url: &str,
    model: Option<&str>,
    messages: Vec<ChatMessage>,
    api_key: Option<String>,
    tools: Option<serde_json::Value>,
    reply_tokens: u32,
) -> impl Stream<Item = Result<ChatStreamItem>> {
    let url = format!("{}/chat/completions", base_url.trim_end_matches('/'));
    // `model: None` means the local llama-server, which must NOT be sent one (see
    // ChatRequest::model) and does understand `cache_prompt`. Passing it
    // explicitly beats inferring "local" from the absence of an API key, which
    // would silently drop the model for any keyless cloud endpoint.
    let is_local = model.is_none();
    let body = ChatRequest {
        model,
        messages: &messages,
        stream: true,
        max_tokens: reply_tokens,
        cache_prompt: is_local,
        tools,
    };

    // `json()` serializes eagerly into the builder, so `body` (and its borrow of
    // `messages`) is no longer needed once the request is built.
    //
    // NOTE: do NOT add reqwest's `.timeout()` here. `.eventsource()` is built
    // synchronously inside the view's `send()` (no Tokio context), and a request
    // timeout arms a Tokio timer at construction → "there is no reactor running"
    // panic. The ollama path can use a timeout because it's polled lazily under
    // spawn_stream_local (which has a Tokio context); the SSE path cannot.
    let client = cloud_http_client();
    let mut request = client.post(url);
    // BYOK: cloud providers (OpenAI / OpenRouter / Groq / HuggingFace …) need the
    // user's key as a bearer token. The local Turbo server ignores auth, so it's
    // only attached when a non-empty key is supplied.
    if let Some(key) = api_key.as_deref().filter(|k| !k.trim().is_empty()) {
        request = request.bearer_auth(key.trim());
    }
    let event_source = request.json(&body).eventsource();

    event_source.filter_map(|event| async move {
        match event {
            Ok(Event::Open) => None,
            Ok(Event::Message(message)) => {
                let data = message.data.trim();
                // OpenAI-style terminator. ollama/llama-server both send this.
                if data == "[DONE]" {
                    return Some(Ok(ChatStreamItem::Done));
                }
                match serde_json::from_str::<ChatChunk>(data) {
                    Ok(chunk) => {
                        // The answer, the thinking and a native tool call are
                        // reported separately, so a thinking model shows progress
                        // without its reasoning being mistaken for the answer, and
                        // a tool call it made on its own channel is not lost.
                        let mut answer = String::new();
                        let mut thinking = String::new();
                        let mut tool_name: Option<String> = None;
                        let mut tool_args = String::new();
                        for choice in chunk.choices {
                            if let Some(content) = choice.delta.content {
                                answer.push_str(&content);
                            }
                            if let Some(reasoning) =
                                choice.delta.reasoning_content.or(choice.delta.reasoning)
                            {
                                thinking.push_str(&reasoning);
                            }
                            for call in choice.delta.tool_calls.into_iter().flatten() {
                                let Some(function) = call.function else {
                                    continue;
                                };
                                if let Some(name) = function.name.filter(|n| !n.is_empty()) {
                                    tool_name = Some(name);
                                }
                                if let Some(arguments) = function.arguments {
                                    tool_args.push_str(&arguments);
                                }
                            }
                        }
                        // A chunk carries exactly one of these in practice, so the
                        // order below only decides an impossible tie. Answer text
                        // wins because it is what the user reads.
                        if !answer.is_empty() {
                            Some(Ok(ChatStreamItem::Token(answer)))
                        } else if tool_name.is_some() || !tool_args.is_empty() {
                            Some(Ok(ChatStreamItem::ToolCall {
                                name: tool_name,
                                arguments: tool_args,
                            }))
                        } else if !thinking.is_empty() {
                            Some(Ok(ChatStreamItem::Reasoning(thinking)))
                        } else {
                            None
                        }
                    }
                    Err(err) => {
                        log::warn!("local chat: skipping malformed chunk: {err}");
                        None
                    }
                }
            }
            // A normal end of stream, NOT a failure. The event source yields this
            // once the server closes the connection, which happens right after
            // `[DONE]` — reporting it put an error banner under every successful
            // answer. Ending quietly is correct; a genuinely empty response is
            // caught by the caller, which knows whether any token arrived.
            Err(EventSourceError::StreamEnded) => None,
            // The server refused the request or answered with something that
            // isn't an event stream (llama-server does this for a prompt that
            // exceeds the context window, for example). Both variants carry the
            // response, so read the body and show what it actually said instead
            // of a bare "Invalid status code: 400".
            Err(EventSourceError::InvalidStatusCode(status, response)) => {
                let body = response.text().await.unwrap_or_default();
                Some(Err(anyhow!(
                    "the model server returned {status}{}",
                    format_server_detail(&body)
                )))
            }
            Err(EventSourceError::InvalidContentType(_, response)) => {
                let body = response.text().await.unwrap_or_default();
                Some(Err(anyhow!(
                    "the model server did not return a stream{}",
                    format_server_detail(&body)
                )))
            }
            Err(err) => {
                // reqwest's Display hides the underlying cause (e.g. "connection
                // refused" / "connection reset"), so walk the source chain and
                // append it — otherwise every failure reads the same useless
                // "error sending request for url".
                use std::error::Error as _;
                let mut detail = String::new();
                let mut source = err.source();
                while let Some(cause) = source {
                    detail.push_str(&format!(": {cause}"));
                    source = cause.source();
                }
                Some(Err(anyhow!("local chat stream error: {err}{detail}")))
            }
        }
    })
}

/// Pull the human-readable part out of an error body. llama-server and the
/// OpenAI-compatible providers answer `{"error":{"message":"…"}}`; anything else
/// is shown verbatim, trimmed so a huge HTML page can't flood the panel.
fn format_server_detail(body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        return String::new();
    }
    let message = serde_json::from_str::<serde_json::Value>(body)
        .ok()
        .and_then(|value| {
            value
                .get("error")
                .and_then(|error| {
                    error
                        .get("message")
                        .and_then(|m| m.as_str())
                        .or_else(|| error.as_str())
                })
                .map(str::to_owned)
        })
        .unwrap_or_else(|| body.to_string());
    if message.is_empty() {
        return String::new();
    }
    const MAX_CHARS: usize = 400;
    let mut shown: String = message.chars().take(MAX_CHARS).collect();
    // Compare CHAR counts of the same string — comparing byte lengths against the
    // raw body marked an untruncated 400-char message as truncated.
    if shown.chars().count() < message.chars().count() {
        shown.push('…');
    }
    format!(": {shown}")
}

#[derive(Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    delta: Option<AnthropicStreamDelta>,
}

#[derive(Deserialize)]
struct AnthropicStreamDelta {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: Option<String>,
}

fn stream_chat_anthropic_sse(
    model: &str,
    messages: Vec<ChatMessage>,
    api_key: String,
) -> impl Stream<Item = Result<ChatStreamItem>> {
    const ANTHROPIC_VERSION: &str = "2023-06-01";
    const ANTHROPIC_MAX_TOKENS: u32 = 2048;

    let mut system_parts = Vec::new();
    let mut anthropic_messages = Vec::new();
    for message in messages {
        match message.role.as_str() {
            "system" => system_parts.push(message.content),
            "user" | "assistant" => anthropic_messages.push(AnthropicInputMessage {
                role: message.role,
                content: message.content,
                images: message.images,
            }),
            _ => anthropic_messages.push(AnthropicInputMessage {
                role: "user".to_string(),
                content: message.content,
                images: message.images,
            }),
        }
    }

    let body = AnthropicChatRequest {
        model,
        max_tokens: ANTHROPIC_MAX_TOKENS,
        messages: &anthropic_messages,
        stream: true,
        system: (!system_parts.is_empty()).then(|| system_parts.join("\n\n")),
    };

    let client = cloud_http_client();
    let event_source = client
        .post(format!(
            "{}/v1/messages",
            CloudProviderKind::Anthropic.base_url()
        ))
        .header("content-type", "application/json")
        .header("anthropic-version", ANTHROPIC_VERSION)
        .header("x-api-key", api_key.trim())
        .json(&body)
        .eventsource();

    event_source.filter_map(|event| async move {
        match event {
            Ok(Event::Open) => None,
            Ok(Event::Message(message)) => {
                let data = message.data.trim();
                if data == "[DONE]" {
                    return Some(Ok(ChatStreamItem::Done));
                }
                match serde_json::from_str::<AnthropicStreamEvent>(data) {
                    Ok(parsed) => match parsed.kind.as_str() {
                        "content_block_delta" => parsed
                            .delta
                            .filter(|delta| delta.kind == "text_delta")
                            .and_then(|delta| delta.text)
                            .filter(|text| !text.is_empty())
                            .map(ChatStreamItem::Token)
                            .map(Ok),
                        "message_stop" => Some(Ok(ChatStreamItem::Done)),
                        _ => None,
                    },
                    Err(err) => {
                        log::warn!("anthropic chat: skipping malformed chunk: {err}");
                        None
                    }
                }
            }
            Err(err) => {
                use std::error::Error as _;
                let mut detail = String::new();
                let mut source = err.source();
                while let Some(cause) = source {
                    detail.push_str(&format!(": {cause}"));
                    source = cause.source();
                }
                Some(Err(anyhow!("cloud chat stream error: {err}{detail}")))
            }
        }
    })
}

// ── local GGUF models ────────────────────────────────────────────────────────
// `/v1/models` only ever knows what `ollama pull` fetched (or, on :11435, the one
// file llama-server already has open), so a GGUF the user imported into Genesi is
// invisible to it. `genesi-ai-turbo` owns that library, so we ask it — the same
// source the AI Mode Monitor uses, which keeps both apps showing one list.
//
// A GGUF is referenced everywhere by the stable string `gguf:<file stem>`; it
// stays a plain String so it flows through the picker and the request path with
// no special casing.

/// True when `model` names a local GGUF rather than an Ollama tag.
pub fn is_gguf_ref(model: &str) -> bool {
    model.starts_with("gguf:") || model.to_ascii_lowercase().ends_with(".gguf")
}

#[derive(Deserialize)]
struct TurboGgufEntry {
    #[serde(default)]
    path: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    params_b: f32,
}

/// The local GGUF library as `(reference, label)` pairs. Returns an empty list
/// when `genesi-ai-turbo` is missing or fails — a machine without it simply has
/// no local library, which is not an error worth showing the user.
pub async fn list_gguf_models() -> Vec<(String, String)> {
    let mut command = command::r#async::Command::new("genesi-ai-turbo");
    command.arg("gguf-list");
    let Ok(output) = command.output().await else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    let entries: Vec<TurboGgufEntry> = match serde_json::from_slice(&output.stdout) {
        Ok(entries) => entries,
        Err(err) => {
            log::warn!("genesi-ai-turbo gguf-list: unparseable output: {err}");
            return Vec::new();
        }
    };
    entries
        .into_iter()
        .filter_map(|entry| {
            let stem = std::path::Path::new(&entry.path)
                .file_stem()?
                .to_string_lossy()
                .into_owned();
            if stem.is_empty() {
                return None;
            }
            let name = if entry.name.trim().is_empty() {
                stem.clone()
            } else {
                entry.name.trim().to_string()
            };
            // "30B", not "30.0B" — but keep a real fraction ("1.5B").
            let size = if entry.params_b > 0.0 {
                let rounded = (entry.params_b * 10.0).round() / 10.0;
                if (rounded.fract()).abs() < f32::EPSILON {
                    format!(" · {}B", rounded as i64)
                } else {
                    format!(" · {rounded:.1}B")
                }
            } else {
                String::new()
            };
            Some((format!("gguf:{stem}"), format!("{name}{size} (GGUF)")))
        })
        .collect()
}

/// Which local endpoint must serve `model`, given the endpoint the UI currently
/// has selected.
///
/// A GGUF can only be loaded by llama-server, so it always overrides the
/// selection. The endpoint field is written from several places (the model
/// picker, the Turbo auto-default, the endpoint cycler), and when it drifted out
/// of step with the chosen model the GGUF reference was posted to Ollama, which
/// replied `model 'gguf:…' not found`.
pub fn transport_for(model: &str, selected: LocalEndpoint) -> LocalEndpoint {
    if is_gguf_ref(model) {
        LocalEndpoint::Turbo
    } else {
        selected
    }
}

/// Make Turbo serve `model`, restarting it if it currently has another one open.
/// Blocks until llama-server answers `/health`. Only meaningful for a GGUF: an
/// Ollama tag is served by Ollama itself and needs nothing here.
///
/// This mirrors what the AI Mode Monitor does when a GGUF is picked. Without it,
/// selecting a local file in Code would post to a server that has a DIFFERENT
/// model loaded (or none at all) and silently return nothing.
pub async fn ensure_turbo_serving(model: &str) -> Result<()> {
    // Its own session on Linux (the shipping target), so the server outlives this
    // request — and Code itself — exactly like the Monitor's `start_new_session`.
    // `spawn()` nulls stdio by default and `kill_on_drop` is false, so dropping
    // the handle below leaves llama-server running either way.
    #[cfg(unix)]
    let mut command = command::r#async::Command::new_with_session("genesi-ai-turbo");
    #[cfg(not(unix))]
    let mut command = command::r#async::Command::new("genesi-ai-turbo");
    command.args(["serve", model, "--no-spec"]);
    // Turbo defaults to a 4K window, which is fine for a chat but tight for an
    // IDE: every turn carries the agent instructions plus up to ~6K characters
    // of file context, and once that overflows the server can accept the request
    // and return nothing. Ask for a roomier window, without overriding a value
    // the user set deliberately.
    if std::env::var_os("GENESI_TURBO_CTX").is_none() {
        command.env("GENESI_TURBO_CTX", "8192");
    }
    let mut child = command.spawn().map_err(|err| {
        anyhow!("couldn't run genesi-ai-turbo (is genesi-ai-mode installed?): {err}")
    })?;

    // `serve` execs llama-server and stays in the foreground, so readiness comes
    // from polling /health — not from waiting on the child. When a healthy server
    // already has this exact model, `serve` reuses it and exits 0 right away.
    for _ in 0..TURBO_LOAD_TIMEOUT_SECS {
        if turbo_health_ok().await {
            return Ok(());
        }
        if let Ok(Some(exit)) = child.try_status() {
            if turbo_health_ok().await {
                return Ok(()); // reused an existing server, then exited
            }
            return Err(anyhow!(
                "genesi-ai-turbo exited ({exit}) without bringing the model up — \
                 run `genesi-ai-turbo serve {model}` in a terminal to see why"
            ));
        }
        warpui::r#async::Timer::after(Duration::from_secs(1)).await;
    }
    Err(anyhow!(
        "{model} took over {TURBO_LOAD_TIMEOUT_SECS}s to load"
    ))
}

/// How long to wait for llama-server to come up with a freshly picked model.
/// A large MoE read from a cold page cache genuinely takes minutes.
const TURBO_LOAD_TIMEOUT_SECS: u32 = 300;

/// Best-effort `GET {base_url}/models` to populate the model picker. Returns the
/// model ids (e.g. `llama3.2:3b`). Errors bubble up so the UI can hint that the
/// local server (ollama) may not be running.
pub async fn list_models(base_url: &str) -> Result<Vec<String>> {
    #[derive(Deserialize)]
    struct ModelsResponse {
        #[serde(default)]
        data: Vec<ModelEntry>,
    }

    #[derive(Deserialize)]
    struct ModelEntry {
        id: String,
    }

    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = http_client::Client::new();
    let response = client
        .get(url)
        .send()
        .await
        .context("failed to reach the local model endpoint")?;
    let parsed: ModelsResponse = response
        .json()
        .await
        .context("failed to parse the local model list")?;
    Ok(parsed.data.into_iter().map(|entry| entry.id).collect())
}

/// genesi-ai-turbo's speculative-decoding server (OpenAI API on :11435).
/// Same payload shape as ollama; faster generation, identical output.
pub const TURBO_LOCAL_BASE_URL: &str = "http://localhost:11435/v1";

/// llama-server's readiness endpoint (root, not under `/v1`). It answers 200
/// once a model is loaded — the reliable liveness signal for Turbo, unlike
/// `/v1/models`. This is what the AI Mode monitor polls too.
pub const TURBO_HEALTH_URL: &str = "http://localhost:11435/health";

/// llama-server's runtime settings, including the context window it was actually
/// started with.
pub const TURBO_PROPS_URL: &str = "http://localhost:11435/props";

/// What to assume when the server won't say — llama.cpp's own default, and the
/// value genesi-ai-turbo used to start with.
pub const ASSUMED_CONTEXT_TOKENS: u32 = 4096;

/// Ask llama-server how big its context window really is.
///
/// Code can ASK for a window when it launches Turbo itself, but `serve` reuses an
/// already-running server, so the request is silently ignored whenever the model
/// was loaded from the Monitor or by hand. The result was a 4K window the agent
/// loop knew nothing about, and a turn that died mid-task with "request (4119
/// tokens) exceeds the available context size (4096 tokens)".
pub async fn turbo_context_size() -> Option<u32> {
    #[derive(Deserialize)]
    struct Props {
        #[serde(default)]
        default_generation_settings: GenerationSettings,
        /// Some builds report it at the top level instead.
        #[serde(default)]
        n_ctx: Option<u32>,
    }

    #[derive(Deserialize, Default)]
    struct GenerationSettings {
        #[serde(default)]
        n_ctx: Option<u32>,
    }

    let client = http_client::Client::new();
    let response = client
        .get(TURBO_PROPS_URL)
        .timeout(Duration::from_secs(3))
        .send()
        .await
        .ok()?;
    if !response.status().is_success() {
        return None;
    }
    let props: Props = response.json().await.ok()?;
    props
        .default_generation_settings
        .n_ctx
        .or(props.n_ctx)
        .filter(|n| *n > 0)
}

/// Rough token count for budgeting. Deliberately a cheap approximation (~4 chars
/// per token) rather than a real tokenizer: the budget only has to keep us
/// clear of the wall, and being a little pessimistic is free while being wrong
/// costs the user their turn.
pub fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Where `genesi-aid` publishes its live status (tmpfs, daemon-written).
pub const AI_MODE_STATE_FILE: &str = "/run/genesi-ai-mode/state.json";
/// User-writable override the daemon reads: `on` | `off` | absent (= auto).
pub const AI_MODE_FORCE_FILE: &str = "/run/genesi-ai-mode/force";

/// The local inference endpoint the chat talks to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalEndpoint {
    /// Ollama's OpenAI-compatible API on :11434 (genesi-ai-mode default).
    Ollama,
    /// genesi-ai-turbo (speculative decoding) on :11435.
    Turbo,
}

impl LocalEndpoint {
    pub fn base_url(self) -> &'static str {
        match self {
            LocalEndpoint::Ollama => DEFAULT_LOCAL_BASE_URL,
            LocalEndpoint::Turbo => TURBO_LOCAL_BASE_URL,
        }
    }

    /// Short label for the endpoint chip.
    pub fn label(self) -> &'static str {
        match self {
            LocalEndpoint::Ollama => "Local",
            LocalEndpoint::Turbo => "Turbo",
        }
    }

    /// The other endpoint (used by the chip that cycles between them).
    pub fn toggled(self) -> Self {
        match self {
            LocalEndpoint::Ollama => LocalEndpoint::Turbo,
            LocalEndpoint::Turbo => LocalEndpoint::Ollama,
        }
    }
}

/// Best-effort liveness probe: succeeds when the endpoint answers `GET /models`.
/// Used to know whether Turbo (:11435) is actually up before offering it.
pub async fn endpoint_available(base_url: &str) -> bool {
    list_models(base_url).await.is_ok()
}

/// Turbo (llama-server) liveness via `GET /health`. llama-server's `/v1/models`
/// isn't a reliable readiness signal, so probe `/health` like the AI Mode
/// monitor does. Returns true when it answers 200.
pub async fn turbo_health_ok() -> bool {
    let client = http_client::Client::new();
    match client
        .get(TURBO_HEALTH_URL)
        .timeout(Duration::from_secs(3))
        .send()
        .await
    {
        Ok(response) => response.status().is_success(),
        Err(_) => false,
    }
}

/// Snapshot of `genesi-aid`'s state, as published in `state.json`. Only the
/// fields the chat panel surfaces are parsed; everything else is ignored.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AiModeState {
    /// Whether AI Mode optimizations are currently applied.
    #[serde(default)]
    pub ai_mode_active: bool,
    /// Auto-detected workload: `active` | `warm` | `idle`.
    #[serde(default)]
    pub activity: String,
    /// User override in effect: `on` | `off` | `auto`.
    #[serde(default)]
    pub force_mode: String,
    /// Effective intensity profile: `max` | `balanced` | `battery` | `auto`.
    #[serde(default)]
    pub profile: String,
    /// Last measured generation speed, when AI Mode is active.
    #[serde(default)]
    pub tokens_per_second: Option<f64>,
}

impl AiModeState {
    /// A compact human-readable status for the header badge.
    pub fn badge_text(&self) -> String {
        let base = if self.force_mode == "on" {
            "AI Mode: on (forced)".to_string()
        } else if self.force_mode == "off" {
            "AI Mode: off (forced)".to_string()
        } else if self.ai_mode_active {
            format!(
                "AI Mode: {}",
                if self.activity.is_empty() {
                    "active"
                } else {
                    &self.activity
                }
            )
        } else {
            "AI Mode: idle".to_string()
        };
        match self.tokens_per_second {
            Some(tps) if tps > 0.0 => format!("{base} · {tps:.0} tok/s"),
            _ => base,
        }
    }
}

/// Read `genesi-aid`'s current state. Returns `None` when the daemon isn't
/// running or its state file isn't present (e.g. on a dev box / non-Genesi host).
pub fn read_ai_mode_state() -> Option<AiModeState> {
    let raw = std::fs::read_to_string(AI_MODE_STATE_FILE).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Set the AI Mode override the daemon honors. `"on"`/`"off"` force the mode;
/// `"auto"` clears the override (removes the file) so the daemon follows its
/// automatic detection again. No-op-safe when the file is already absent.
pub fn set_ai_mode_force(value: &str) -> Result<()> {
    match value {
        "auto" => match std::fs::remove_file(AI_MODE_FORCE_FILE) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(anyhow!("failed to clear AI Mode override: {e}")),
        },
        "on" | "off" => std::fs::write(AI_MODE_FORCE_FILE, value)
            .map_err(|e| anyhow!("failed to set AI Mode override: {e}")),
        other => Err(anyhow!("invalid AI Mode force value: {other}")),
    }
}

// ── BYOK: user-configured cloud provider ──────────────────────────────────────

/// A user-supplied cloud provider (Bring Your Own Key). Any OpenAI-compatible
/// endpoint works. Persisted to `~/.config/genesi-code/cloud.json`. Local stays
/// the default — this is purely opt-in, for a bigger/faster model than the
/// machine can run (or a frontier model for hard tasks). Stored on-device only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CloudProviderKind {
    HuggingFace,
    OpenAI,
    Groq,
    Anthropic,
    Gemini,
}

impl Default for CloudProviderKind {
    fn default() -> Self {
        Self::HuggingFace
    }
}

impl CloudProviderKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::HuggingFace => "Hugging Face",
            Self::OpenAI => "OpenAI",
            Self::Groq => "Groq",
            Self::Anthropic => "Anthropic",
            Self::Gemini => "Google Gemini",
        }
    }

    pub fn base_url(self) -> &'static str {
        match self {
            Self::HuggingFace => "https://router.huggingface.co/v1",
            Self::OpenAI => "https://api.openai.com/v1",
            Self::Groq => "https://api.groq.com/openai/v1",
            Self::Anthropic => "https://api.anthropic.com",
            Self::Gemini => "https://generativelanguage.googleapis.com/v1beta/openai",
        }
    }

    pub fn default_model(self) -> &'static str {
        match self {
            Self::HuggingFace => "openai/gpt-oss-120b:cerebras",
            Self::OpenAI => "gpt-4o-mini",
            Self::Groq => "llama-3.3-70b-versatile",
            Self::Anthropic => "claude-sonnet-4-6",
            Self::Gemini => "gemini-3.5-flash",
        }
    }

    pub fn suggested_models(self) -> &'static [&'static str] {
        match self {
            Self::HuggingFace => &[
                "openai/gpt-oss-120b:cerebras",
                "Qwen/Qwen3-Coder-480B-A35B-Instruct",
                "Qwen/Qwen2.5-Coder-32B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct",
            ],
            Self::OpenAI => &["gpt-5.4", "gpt-4o-mini"],
            Self::Groq => &[
                "llama-3.3-70b-versatile",
                "openai/gpt-oss-120b",
                "moonshotai/kimi-k2-instruct",
                "qwen/qwen3-32b",
                "deepseek-r1-distill-llama-70b",
            ],
            Self::Anthropic => &[
                "claude-fable-5",
                "claude-opus-4-8",
                "claude-sonnet-4-6",
                "claude-haiku-4-5",
            ],
            Self::Gemini => &["gemini-3.5-flash"],
        }
    }

    pub fn from_legacy(label: &str, base_url: &str) -> Self {
        let normalized = format!("{label}\n{base_url}").to_ascii_lowercase();
        if normalized.contains("anthropic") || normalized.contains("claude") {
            Self::Anthropic
        } else if normalized.contains("gemini")
            || normalized.contains("generativelanguage.googleapis.com")
        {
            Self::Gemini
        } else if normalized.contains("groq") || normalized.contains("api.groq.com") {
            Self::Groq
        } else if normalized.contains("openai") || normalized.contains("api.openai.com") {
            Self::OpenAI
        } else {
            Self::HuggingFace
        }
    }
}

pub fn cloud_presets() -> &'static [CloudProviderKind] {
    &[
        CloudProviderKind::HuggingFace,
        CloudProviderKind::OpenAI,
        CloudProviderKind::Groq,
        CloudProviderKind::Anthropic,
        CloudProviderKind::Gemini,
    ]
}

pub const CLOUD_KEYS_STORAGE_KEY: &str = "GenesiCodeCloudApiKeys";

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloudKeyStore {
    #[serde(default)]
    hugging_face: String,
    #[serde(default)]
    openai: String,
    #[serde(default)]
    groq: String,
    #[serde(default)]
    anthropic: String,
    #[serde(default)]
    gemini: String,
}

impl CloudKeyStore {
    pub fn get(&self, provider: CloudProviderKind) -> &str {
        match provider {
            CloudProviderKind::HuggingFace => &self.hugging_face,
            CloudProviderKind::OpenAI => &self.openai,
            CloudProviderKind::Groq => &self.groq,
            CloudProviderKind::Anthropic => &self.anthropic,
            CloudProviderKind::Gemini => &self.gemini,
        }
    }

    pub fn set(&mut self, provider: CloudProviderKind, value: String) {
        match provider {
            CloudProviderKind::HuggingFace => self.hugging_face = value,
            CloudProviderKind::OpenAI => self.openai = value,
            CloudProviderKind::Groq => self.groq = value,
            CloudProviderKind::Anthropic => self.anthropic = value,
            CloudProviderKind::Gemini => self.gemini = value,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.hugging_face.trim().is_empty()
            && self.openai.trim().is_empty()
            && self.groq.trim().is_empty()
            && self.anthropic.trim().is_empty()
            && self.gemini.trim().is_empty()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Selected cloud provider.
    #[serde(default)]
    pub provider: CloudProviderKind,
    /// The model id to request, e.g. `gpt-4o-mini` or a Hugging Face repo id.
    #[serde(default)]
    pub model: String,
}

impl CloudConfig {
    pub fn with_defaults(mut self) -> Self {
        if self.model.trim().is_empty() {
            self.model = self.provider.default_model().to_string();
        }
        self
    }

    /// Whether this config can actually be used (has a key + model).
    pub fn is_ready(&self, api_key: &str) -> bool {
        !self.model.trim().is_empty() && !api_key.trim().is_empty()
    }
}

/// Path to the on-device BYOK config (`$XDG_CONFIG_HOME` or `~/.config`).
pub fn cloud_config_path() -> Option<PathBuf> {
    let dir = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(dir.join("genesi-code").join("cloud.json"))
}

/// Load the saved cloud provider, if any.
pub fn load_cloud_config() -> Option<CloudConfig> {
    let path = cloud_config_path()?;
    let raw = std::fs::read_to_string(path).ok()?;
    serde_json::from_str::<CloudConfig>(&raw)
        .map(CloudConfig::with_defaults)
        .or_else(|_| {
            serde_json::from_str::<LegacyCloudConfig>(&raw).map(|legacy| CloudConfig {
                provider: CloudProviderKind::from_legacy(&legacy.label, &legacy.base_url),
                model: if legacy.model.trim().is_empty() {
                    CloudProviderKind::from_legacy(&legacy.label, &legacy.base_url)
                        .default_model()
                        .to_string()
                } else {
                    legacy.model
                },
            })
        })
        .ok()
}

#[derive(Debug, Clone, Default, Deserialize)]
struct LegacyCloudConfig {
    #[serde(default)]
    label: String,
    #[serde(default)]
    base_url: String,
    #[serde(default)]
    api_key: String,
    #[serde(default)]
    model: String,
}

/// Pull a plaintext key from the legacy config file so the UI can migrate it to
/// secure storage on first launch after upgrading.
pub fn load_legacy_cloud_key() -> Option<(CloudProviderKind, String)> {
    let path = cloud_config_path()?;
    let raw = std::fs::read_to_string(path).ok()?;
    let legacy = serde_json::from_str::<LegacyCloudConfig>(&raw).ok()?;
    let key = legacy.api_key.trim().to_string();
    if key.is_empty() {
        return None;
    }
    Some((
        CloudProviderKind::from_legacy(&legacy.label, &legacy.base_url),
        key,
    ))
}

/// Persist the cloud metadata to disk (creating the config directory). API
/// keys are deliberately not stored here; they live in secure storage.
pub fn save_cloud_config(cfg: &CloudConfig) -> Result<()> {
    let path = cloud_config_path().ok_or_else(|| anyhow!("no config directory (HOME unset)"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| anyhow!("failed to create config dir: {e}"))?;
    }
    let json = serde_json::to_string_pretty(&cfg.clone().with_defaults())
        .map_err(|e| anyhow!("failed to serialize config: {e}"))?;
    std::fs::write(&path, json).map_err(|e| anyhow!("failed to write cloud config: {e}"))?;
    Ok(())
}

#[cfg(test)]
#[path = "local_chat_tests.rs"]
mod tests;
