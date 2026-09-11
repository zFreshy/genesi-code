//! Login-free local AI chat panel.
//!
//! A small side panel that talks to a local OpenAI-compatible endpoint via
//! [`super::local_chat`] — ollama on `:11434` ("Local") or `genesi-ai-turbo` on
//! `:11435` ("Turbo"). It also surfaces, and lets you drive, `genesi-ai-mode`'s
//! AI Mode (the daemon that tunes the box for inference) so the whole local-AI
//! story lives in one place: no account, no cloud.
#![allow(dead_code)]

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::Result;
use markdown_parser::{parse_markdown, FormattedText, FormattedTextFragment, FormattedTextLine};
use parking_lot::RwLock;
use pathfinder_geometry::rect::RectF;
use serde::{Deserialize, Serialize};
use similar::{Algorithm, ChangeTag, TextDiff};
use warp_core::ui::color::blend::Blend;
use warp_core::ui::icons::Icon as CoreIcon;
use warp_core::ui::theme::Fill as ThemeFill;
use warpui::clipboard::ClipboardContent;
use warpui::color::ColorU;
use warpui::elements::{
    Border, ChildAnchor, Clipped, ClippedScrollStateHandle, ClippedScrollable, ConstrainedBox,
    Container, CornerRadius, CrossAxisAlignment, DispatchEventResult, DropTarget, DropTargetData,
    Element, Empty, EventHandler, Expanded, Fill, Flex, FormattedTextElement, Icon,
    MainAxisAlignment, MainAxisSize, OffsetPositioning, ParentAnchor, ParentElement,
    ParentOffsetBounds, Radius, ScrollbarWidth, SelectableArea, SelectionHandle, Shrinkable, Stack,
};
use warpui::geometry::vector::{vec2f, Vector2F};
use warpui::keymap::FixedBinding;
use warpui::platform::FilePickerConfiguration;
use warpui::presenter::ChildView;
use warpui::ui_components::components::{Coords, UiComponent, UiComponentStyles};
use warpui::units::Pixels;
use warpui::{
    AppContext, Entity, FocusContext, SingletonEntity, TypedActionView, View, ViewContext,
    ViewHandle, WeakViewHandle,
};
use warpui_extras::secure_storage::AppContextExt as _;

use super::api_probe::{
    build_url, method_takes_body, scan_ports, send_request, survey_project, ProbePort,
    ProbeRequest, ProbeResponse, ProbeRoute, ProbeSurvey, PROBE_METHODS,
};
use super::lenses::{architecture, performance, ArchitectureLens, PerformanceLens, Severity};
use super::local_agent::{self, AgentTool, MAX_AGENT_STEPS};
use super::local_chat::{
    cloud_presets, ensure_turbo_serving, estimate_tokens, is_gguf_ref, is_tools_unsupported_error,
    list_gguf_models, list_models, load_cloud_config, load_legacy_cloud_key, model_supports_vision,
    read_ai_mode_state, reply_budget, save_cloud_config, set_ai_mode_force, stream_chat,
    stream_chat_cloud, transport_for, turbo_context_size, turbo_health_ok, AiModeState,
    AttachmentKind, ChatAttachment, ChatMessage, ChatStreamItem, CloudConfig, CloudKeyStore,
    CloudProviderKind, CodeContext, LocalEndpoint, ASSUMED_CONTEXT_TOKENS, CLOUD_KEYS_STORAGE_KEY,
    DEFAULT_LOCAL_BASE_URL,
};
use super::project_canvas::{
    analyze_project, CanvasEdgeKind, CanvasNode, CanvasNodeKind, ProjectCanvasGraph, ProjectKind,
};
use super::project_canvas_view::{
    CanvasViewport, ProjectGraphCanvas, ProjectGraphColors, ProjectGraphMinimap,
    ProjectGraphNodeElement, FORGE_NODE_HEIGHT, FORGE_NODE_WIDTH,
};
use crate::appearance::Appearance;
use crate::code::file_tree::FileTreeView;
use crate::code::local_code_editor::LocalCodeEditorView;
use crate::editor::{EditorOptions, EditorView, Event as EditorEvent, TextOptions};
use crate::settings_view::SettingsSection;
use crate::util::bindings::CustomAction;
use crate::view_components::{SubmittableTextInput, SubmittableTextInputEvent};
use crate::workspace::WorkspaceAction;

const TITLE_FONT_SIZE: f32 = 15.;
const CHIP_FONT_SIZE: f32 = 11.;
const BODY_FONT_SIZE: f32 = 13.;
/// Slightly smaller monospace size for tool output / terminal blocks.
const MONO_FONT_SIZE: f32 = 12.;
const PANEL_PADDING: f32 = 8.;
const MODEL_LABEL_MAX_CHARS: usize = 34;
const VIBE_COLUMN_WIDTH: f32 = 760.;
const MEMPALACE_STORAGE_KEY: &str = "GenesiCodeMempalaceV1";
/// A large scroll target; `ClippedScrollable::after_layout` clamps it to the
/// real bottom, so this reliably pins the transcript to the latest message.
const SCROLL_TO_BOTTOM: f32 = 1.0e7;
const CANVAS_DRAG_FRAME_INTERVAL: Duration = Duration::from_millis(16);
const CANVAS_DRAG_MIN_DISTANCE: f32 = 3.;

/// How many past Probe requests stay replayable. Deep enough to get back to the
/// one before the one you broke, shallow enough that the list stays scannable.
const PROBE_HISTORY_LIMIT: usize = 20;

/// Turn the Canvas's body hint (`"JSON fields: name, email"`) into a JSON
/// skeleton. Any other shape of hint — a type name, a note that the schema is
/// only knowable at runtime — has no fields to lay out, so the body is left
/// empty rather than filled with a guess.
fn probe_body_template(hint: &str) -> Option<String> {
    let fields = hint
        .strip_prefix("JSON fields: ")?
        .split(',')
        .map(str::trim)
        .filter(|field| !field.is_empty())
        .collect::<Vec<_>>();
    if fields.is_empty() {
        return None;
    }
    let body = fields
        .iter()
        .map(|field| format!("  \"{field}\": \"\""))
        .collect::<Vec<_>>()
        .join(",\n");
    Some(format!("{{\n{body}\n}}"))
}

/// The path (and query) of a URL that may have been typed by hand, so switching
/// ports keeps the request pointed at the same route.
fn probe_path_of(url: &str) -> String {
    let url = url.trim();
    let after_scheme = url.split_once("://").map(|(_, rest)| rest).unwrap_or(url);
    match after_scheme.find('/') {
        Some(slash) => after_scheme[slash..].to_string(),
        None => "/".to_string(),
    }
}

const SYSTEM_PROMPT: &str =
    "You are a helpful AI assistant running locally on Genesi OS. Be concise.";

/// How many times one turn may re-prompt a model that produced only reasoning.
/// A thinking model sometimes plans its next step and stops without ever opening
/// its answer channel; asking again costs one round trip and usually gets the
/// tool call. Bounded so a model that ONLY ever thinks can't spin forever.
const MAX_AGENT_NUDGES: u32 = 2;

/// Tokens held back from the prompt so the model can actually reply.
const REPLY_RESERVE_TOKENS: usize = 768;

/// A trimmed message keeps at least this much, so shrinking never reduces a tool
/// result to something meaningless — better to drop a message whole than to feed
/// the model a stub it will misread.
const MIN_KEPT_MESSAGE_TOKENS: usize = 96;

/// Truncate `text` to roughly `max_tokens`, keeping the START (where a tool
/// result says what it is) and marking the cut so the model knows it is partial.
fn clamp_to_tokens(text: &str, max_tokens: usize) -> String {
    let max_chars = max_tokens.saturating_mul(4);
    if text.len() <= max_chars {
        return text.to_string();
    }
    let mut out: String = text.chars().take(max_chars).collect();
    out.push_str("\n… [truncated to fit the model's context window]");
    out
}

/// Marks the AI panel's compose box as a place the file tree can drop onto, so a
/// file can be dragged straight into the conversation as a reference — the same
/// gesture that already drops a file onto the terminal input.
#[derive(Debug, Clone)]
pub struct LocalAiDropTargetData {
    panel: WeakViewHandle<LocalAiChatView>,
}

impl LocalAiDropTargetData {
    pub fn panel(&self) -> WeakViewHandle<LocalAiChatView> {
        self.panel.clone()
    }
}

impl DropTargetData for LocalAiDropTargetData {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Whether the active server accepts the OpenAI `tools` field. Learned at
/// runtime rather than configured: llama-server only takes it when built/started
/// with `--jinja`, and there is no capability endpoint to ask.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeToolSupport {
    /// Not yet established — send tools and find out.
    Untried,
    /// A request with tools was accepted.
    Supported,
    /// The server refused the field; stop sending it for this session.
    Unsupported,
}

pub fn init(app: &mut AppContext) {
    use warpui::keymap::macros::*;

    app.register_fixed_bindings([FixedBinding::custom(
        CustomAction::Copy,
        LocalAiChatAction::CopySelectedText,
        "Copy",
        id!(LocalAiChatView::ui_name()) & !id!("IMEOpen"),
    )]);
}

/// The Genesi brand green, used as the panel's accent.
fn genesi_green() -> ColorU {
    ColorU::new(15, 143, 106, 255)
}

/// A very translucent Genesi green — for the subtle fill behind the user's
/// messages and the active control chips.
fn green_tint() -> ColorU {
    ColorU::new(15, 143, 106, 38)
}

/// A semi-opaque Genesi green — for the borders of active controls and the
/// user-message accent.
fn green_soft() -> ColorU {
    ColorU::new(15, 143, 106, 130)
}

fn genesi_panel_surface() -> ColorU {
    ColorU::new(20, 21, 23, 245)
}

fn genesi_card_surface() -> ColorU {
    ColorU::new(31, 32, 35, 245)
}

/// The compose box. Clearly lighter than the panel fill it sits on
/// (`genesi_shell_panel_surface`, rgb 30/31/34) so it reads as a raised card
/// without needing a border.
fn genesi_compose_surface() -> ColorU {
    ColorU::new(46, 48, 53, 255)
}

fn genesi_subtle_border() -> ColorU {
    ColorU::new(255, 255, 255, 24)
}

fn truncate_middle(value: &str, max_chars: usize) -> String {
    let len = value.chars().count();
    if len <= max_chars {
        return value.to_string();
    }

    let head_len = (max_chars.saturating_sub(3) + 1) / 2;
    let tail_len = max_chars.saturating_sub(3 + head_len);
    let head: String = value.chars().take(head_len).collect();
    let tail: String = value
        .chars()
        .rev()
        .take(tail_len)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("{head}...{tail}")
}

fn ai_mode_short_label(state: Option<&AiModeState>) -> String {
    match state {
        Some(state) => format!("Mode {}", state.force_mode),
        None => "Mode n/a".to_string(),
    }
}

/// What the compose box is currently capturing: a chat prompt, or a one-shot
/// value for the BYOK cloud provider (its API key or model id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputMode {
    Chat,
    CloudKey,
    CloudModel,
}

/// Who authored a transcript entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum ChatRole {
    User,
    /// The model's final, human-facing answer — rendered as markdown.
    Assistant,
    /// The model's reasoning for an agent step (the text it streamed before a
    /// tool call). Shown as a collapsible "💭 Thought" so the raw tool markup
    /// never clutters the transcript.
    Thought,
    /// A read-only agent tool step (read_file / list_files / grep / edit_file) —
    /// a collapsible one-line summary with the result tucked underneath.
    Tool,
    /// A `run_command` step — rendered as a terminal block (`$ cmd` + output).
    Command,
}

/// How an agent step (Tool / Command / Thought) is doing, for its status icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum StepStatus {
    Running,
    Ok,
    Error,
    Denied,
}

/// One line in the transcript. The assistant's text grows as tokens stream in.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatEntry {
    role: ChatRole,
    text: String,
    /// For user turns: a short label of the code context attached (if any), e.g.
    /// `foo.rs · lines 12-40`. Shown faintly under the message.
    context_label: Option<String>,
    /// For Tool steps: the one-line header (e.g. `read_file src/main.rs`).
    tool_title: Option<String>,
    /// For Command steps: the shell command line that was run.
    command: Option<String>,
    /// Collapsed state for the collapsible step kinds (Thought / Tool / Command).
    collapsed: bool,
    /// Status used to pick the step's icon/color.
    status: StepStatus,
    /// For a file write/edit Tool step: `(added, removed)` line counts, rendered
    /// as a `+N −N` diff card. `None` for every other step.
    diff_stat: Option<(u32, u32)>,
    diff_preview: Option<Vec<DiffPreviewLine>>,
}

impl ChatEntry {
    /// A plain prose entry (User / Assistant).
    fn prose(role: ChatRole, text: String, context_label: Option<String>) -> Self {
        Self {
            role,
            text,
            context_label,
            tool_title: None,
            command: None,
            collapsed: false,
            status: StepStatus::Ok,
            diff_stat: None,
            diff_preview: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum DiffPreviewLineKind {
    Context,
    Added,
    Removed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DiffPreviewLine {
    old_line: Option<usize>,
    new_line: Option<usize>,
    text: String,
    kind: DiffPreviewLineKind,
}

/// A file the agent created or edited this turn, tracked so the review bar can
/// summarize the changes and undo them (restore the captured original, or delete
/// a file that didn't exist before).
struct PendingEdit {
    /// Project-relative path.
    path: String,
    added: u32,
    removed: u32,
    /// The file's content before the edit, or `None` if the file was created.
    original: Option<String>,
    diff_preview: Vec<DiffPreviewLine>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenesiSideTool {
    Review,
    Canvas,
    Lenses,
}

#[derive(Debug)]
enum ProjectCanvasState {
    NoProject,
    Loading(PathBuf),
    Ready(Arc<ProjectCanvasGraph>),
    Error(String),
}

#[derive(Debug, Clone)]
enum ProjectCanvasDragState {
    Pan {
        pointer: Vector2F,
        pan: Vector2F,
    },
    Node {
        id: String,
        pointer: Vector2F,
        position: Vector2F,
    },
}

/// What Genesi Probe knows about the project it is pointed at.
#[derive(Debug)]
enum ProbeState {
    NoProject,
    Loading(PathBuf),
    Ready(Arc<ProbeSurvey>),
    Error(String),
}

/// Where the current request is. Separate from [`ProbeState`] because a failed
/// request says nothing about the survey — the routes are still valid, the
/// server just isn't up.
#[derive(Debug)]
enum ProbeResponseState {
    Idle,
    InFlight,
    /// Boxed: [`ProbeResponse`] carries the whole body, and this enum is a field
    /// on a view that is moved around on every update.
    Ready(Box<ProbeResponse>),
    Failed(String),
}

/// A request the user already sent, kept so it can be fired again unchanged.
#[derive(Debug, Clone)]
struct ProbeHistoryEntry {
    method: String,
    url: String,
    headers: String,
    body: String,
    /// `None` when the request never reached a server.
    status: Option<u16>,
    elapsed_ms: u128,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct PersistedLocalChat {
    id: String,
    #[serde(default)]
    messages: Vec<ChatEntry>,
    #[serde(default)]
    agent_messages: Vec<ChatMessage>,
    #[serde(default)]
    updated_at_ms: u64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct MempalaceState {
    #[serde(default)]
    active_chat_id: String,
    #[serde(default)]
    chats: Vec<PersistedLocalChat>,
    /// The working mode, so a deliberate choice survives a restart. `None` is a
    /// profile written before this existed, which falls back to the default.
    #[serde(default)]
    chat_mode: Option<ChatMode>,
}

/// Count lines for a diff stat: 0 for empty, else the number of lines (a
/// trailing newline doesn't add a phantom empty line).
fn count_lines(s: &str) -> u32 {
    if s.is_empty() {
        0
    } else {
        s.lines().count() as u32
    }
}

fn build_diff_preview(original: &str, current: &str) -> Vec<DiffPreviewLine> {
    let diff = TextDiff::configure()
        .algorithm(Algorithm::Patience)
        .diff_lines(original, current);

    let mut old_line = 1usize;
    let mut new_line = 1usize;
    let mut lines = Vec::new();

    for change in diff.iter_all_changes() {
        let kind = match change.tag() {
            ChangeTag::Equal => DiffPreviewLineKind::Context,
            ChangeTag::Delete => DiffPreviewLineKind::Removed,
            ChangeTag::Insert => DiffPreviewLineKind::Added,
        };

        for raw_line in change.value().lines() {
            let (old_number, new_number) = match kind {
                DiffPreviewLineKind::Context => {
                    let nums = (Some(old_line), Some(new_line));
                    old_line += 1;
                    new_line += 1;
                    nums
                }
                DiffPreviewLineKind::Removed => {
                    let nums = (Some(old_line), None);
                    old_line += 1;
                    nums
                }
                DiffPreviewLineKind::Added => {
                    let nums = (None, Some(new_line));
                    new_line += 1;
                    nums
                }
            };

            lines.push(DiffPreviewLine {
                old_line: old_number,
                new_line: new_number,
                text: raw_line.to_string(),
                kind: kind.clone(),
            });
        }
    }

    lines
}

/// Events emitted to the workspace.
pub enum LocalAiChatEvent {
    /// Close the panel.
    ClosePanel,
    /// Open the app's native code review / diff panel for the active repo.
    OpenDiff,
    /// The user submitted a prompt; the workspace attaches fresh file context
    /// and calls back into [`LocalAiChatView::send_with_context`]. Routing this
    /// through the workspace is what gives the panel workspace awareness.
    SubmitPrompt(String),
    /// The persisted chat list or active chat changed.
    StateChanged,
}

#[derive(Debug, Clone)]
pub struct LocalChatSummary {
    pub id: String,
    pub title: String,
    pub is_active: bool,
}

/// How the assistant is allowed to work. One selector replaces what used to be
/// two independent chips (Agent on/off and AUTO on/off), which could express the
/// same three useful states plus a meaningless fourth (AUTO while not an agent).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChatMode {
    /// Answers questions only; no tools, nothing touched.
    Chat,
    /// Uses tools, but asks before anything that changes the machine.
    Build,
    /// Uses tools and runs them without asking.
    Auto,
}

impl ChatMode {
    fn label(self) -> &'static str {
        match self {
            ChatMode::Chat => "Chat",
            ChatMode::Build => "Build",
            ChatMode::Auto => "Auto",
        }
    }

    fn description(self) -> &'static str {
        match self {
            ChatMode::Chat => "Answer only — no files or commands touched",
            ChatMode::Build => "Reads and edits, asking before each change",
            ChatMode::Auto => "Reads and edits without asking",
        }
    }
}

/// Which half of the model picker is showing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPickerTab {
    /// Models running on this machine (ollama tags and local GGUF files).
    Local,
    /// A cloud provider used with the user's own API key.
    Cloud,
}

/// Click actions dispatched by the header chips.
#[derive(Debug, Clone)]
pub enum LocalAiChatAction {
    /// Copy the text currently selected in the transcript.
    CopySelectedText,
    /// Move keyboard focus from the prompt editor to the transcript selection.
    FocusTranscript,
    /// Switch between the Local (ollama) and Turbo endpoints.
    CycleEndpoint,
    /// Advance to the next available model.
    CycleModel,
    /// Cycle the AI Mode override: auto -> on -> off -> auto.
    CycleAiMode,
    /// Re-probe models, endpoints, and AI Mode state.
    Refresh,
    /// Clear the transcript.
    Clear,
    /// Toggle auto-attaching the focused file as context.
    ToggleAttachContext,
    /// Open the OS file picker and attach whatever the user chooses.
    PickAttachments,
    /// Pin these paths to the next message (from the picker's callback).
    AttachPaths(Vec<PathBuf>),
    /// Drop the attachment at this index from the next message.
    RemoveAttachment(usize),
    /// Toggle agent mode (the model can read the project via tools).
    ToggleAgent,
    /// Toggle AUTO: run agent commands/edits without per-action approval.
    ToggleAuto,
    /// Open/close the AI model picker popup (above the compose box).
    ToggleModelPicker,
    /// Open/close the mode picker (Chat / Build / Auto).
    ToggleModePicker,
    /// Select a working mode. This is the single control that replaced the
    /// separate Agent and AUTO chips; it sets the same two flags underneath.
    SetChatMode(ChatMode),
    /// Show only local models, or only cloud providers, in the model picker.
    SetPickerTab(ModelPickerTab),
    /// Pick the local model at this index from the picker, and use the local
    /// (ollama) endpoint.
    PickModel(usize),
    /// Turn the Turbo (full-GPU) endpoint on or off.
    ToggleTurbo,
    /// Accept the agent's file changes and dismiss the review bar.
    KeepEdits,
    /// Undo the agent's file changes (restore each captured original).
    UndoEdits,
    /// Open the workspace diff / review panel for a transcript file card.
    OpenDiff(usize),
    /// Expand/collapse the modified-files review browser.
    ToggleReviewExpanded,
    /// Focus a single file inside the expanded review browser.
    SelectReviewFile(String),
    /// Approve the pending side-effecting tool and run it.
    ApproveTool,
    /// Deny the pending side-effecting tool.
    DenyTool,
    /// Interrupt the in-flight generation / agent loop.
    Stop,
    /// Expand/collapse the transcript entry at this index (thought/tool/command).
    ToggleCollapse(usize),
    /// Expand/collapse the "Ran N commands" run that starts at this index.
    ToggleStepGroup(usize),
    /// Open/close the saved-chat list.
    ToggleChatPicker,
    /// Switch to a saved chat.
    PickChat(String),
    /// Delete a saved chat.
    DeleteChat(String),
    /// Cycle the BYOK cloud provider preset (HuggingFace / OpenAI / …).
    CycleProvider,
    /// Select a specific cloud provider from the picker.
    SelectCloudProvider(CloudProviderKind),
    /// Pick a suggested cloud model for the active provider.
    PickCloudModel(String),
    /// Capture the cloud provider's API key in the compose box.
    SetKey,
    /// Capture the cloud provider's model id in the compose box.
    SetModel,
    /// Clear the active conversation and start a fresh one.
    NewChat,
    /// Submit whatever is currently typed into the compose input.
    SubmitPromptInput,
    /// Drop a starter prompt into the compose box and focus it. The zero-state
    /// cards are the only thing on screen before a conversation exists, so they
    /// may as well start one instead of describing that one could be started.
    StartPrompt(&'static str),
    ToggleSoundscape,
    CycleSoundscape,
    ToggleKeyboardAsmr,
    CycleKeyboardSwitch,
}

pub struct LocalAiChatView {
    /// Handle to this view, so the compose box can advertise itself as a drop
    /// target (render only sees `&AppContext`, which can't produce one).
    weak_handle: WeakViewHandle<Self>,
    input: ViewHandle<SubmittableTextInput>,
    messages: Vec<ChatEntry>,
    active_chat_id: String,
    chats: Vec<PersistedLocalChat>,
    transcript_scroll: ClippedScrollStateHandle,
    transcript_selection: SelectionHandle,
    selected_transcript_text: Arc<RwLock<Option<String>>>,
    review_sidebar_scroll: ClippedScrollStateHandle,
    vibe_mode: bool,

    endpoint: LocalEndpoint,
    turbo_available: bool,
    /// Set once the user picks an endpoint by hand, so the Turbo auto-default
    /// (see [`Self::refresh_models`]) never overrides a deliberate choice.
    endpoint_user_chosen: bool,
    /// Every selectable model as a raw reference: ollama tags, plus local GGUFs
    /// as the stable `gguf:<file stem>` form that genesi-ai-turbo resolves.
    models: Vec<String>,
    /// Display names for the `gguf:` entries above (reference -> friendly name),
    /// so the picker can show "Qwen3 30B A3B · 30B (GGUF)" while the VALUE stays
    /// the reference the backend needs.
    gguf_labels: HashMap<String, String>,
    /// The GGUF currently being loaded into llama-server, if any. Drives the
    /// "loading…" hint and stops a second pick from racing the first.
    preparing_model: Option<String>,
    /// Whether the Chat/Build/Auto popup is showing.
    mode_picker_open: bool,
    /// Which half of the model picker is showing (local vs cloud).
    picker_tab: ModelPickerTab,
    selected_model: Option<usize>,
    ai_mode: Option<AiModeState>,

    in_flight: bool,
    /// Corrections the user typed WHILE a turn was running, waiting to be fed
    /// into it. Before this existed a mid-turn submit was dropped on the floor
    /// by the `in_flight` guard in `send_with_context`, so the only way to
    /// redirect the model was to stop it and start over.
    steering: Vec<String>,
    error: Option<String>,
    /// Monotonic id of the current generation. Stop bumps it so stale stream
    /// callbacks (from a turn the user interrupted) no-op instead of writing
    /// into a fresh turn — the streams are detached and can't be cancelled
    /// directly, so we guard their callbacks by id.
    current_turn: u64,

    /// When on, each send attaches the focused code editor's file (or selection)
    /// as context — like a normal AI IDE.
    attach_context: bool,

    /// Files the user pinned to the next message by hand: the paperclip picker,
    /// or a Ctrl+V of files copied in the file manager. Separate from
    /// `attach_context`, which is the editor's own file following the cursor.
    /// Cleared once the message they were attached to is sent.
    attachments: Vec<ChatAttachment>,

    /// When on, the model runs as a codebase agent: it can read the project via
    /// tools (read_file/list_files/grep) before answering.
    agent_mode: bool,

    // ── agent-loop state (only meaningful while an agent turn is in flight) ──
    /// Project root the agent's tools resolve paths against.
    agent_root: Option<PathBuf>,
    /// The running conversation sent to the model (diverges from the visible
    /// transcript: it carries raw tool calls and tool results).
    agent_messages: Vec<ChatMessage>,
    /// Model name pinned for the duration of the agent turn.
    agent_model: String,
    /// How many tool steps this turn has taken (bounded by [`MAX_AGENT_STEPS`]).
    agent_step: u32,
    /// Accumulates the current step's streamed tokens until it completes.
    agent_step_buffer: String,
    /// Accumulates the current step's REASONING (a thinking model's private
    /// analysis). Deliberately not part of `agent_step_buffer` — see
    /// [`ChatStreamItem::Reasoning`] — but kept because it is the only thing a
    /// harmony model produces when it never reaches its `final` channel, and the
    /// step-end fallback reads it rather than failing the turn.
    agent_reasoning_buffer: String,
    /// A native tool call the model made on its own channel, accumulated across
    /// stream fragments as `(name, arguments-json)`.
    agent_native_call: Option<(Option<String>, String)>,
    /// How many times this turn has re-prompted a model that answered only in its
    /// reasoning channel. Bounded by [`MAX_AGENT_NUDGES`] so a model that always
    /// thinks and never speaks can't loop forever.
    agent_nudges: u32,
    /// Whether this session's server takes the OpenAI `tools` field.
    native_tools: NativeToolSupport,
    /// The context window llama-server is ACTUALLY running with, read from
    /// `/props`. `None` until probed; see [`Self::fit_to_context`].
    server_context: Option<u32>,
    /// Summary of the tool currently running, used to render its step.
    agent_tool_summary: String,

    /// When on, the agent runs commands/edits without asking. Off = approve each.
    auto_approve: bool,
    /// A side-effecting tool waiting for the user's Allow/Deny.
    pending_tool: Option<AgentTool>,

    // ── BYOK: optional cloud provider (off by default; local stays the default) ──
    /// The user's saved cloud provider (metadata only; keys live in secure storage).
    cloud: CloudConfig,
    /// Per-provider API keys stored in platform secure storage and mirrored in memory.
    cloud_keys: CloudKeyStore,
    /// When on, prompts go to the cloud provider instead of Local/Turbo.
    cloud_active: bool,
    /// What the compose box's next submit means (a prompt, or a key/model value).
    input_mode: InputMode,

    /// Whether the click-to-open AI model picker (the popup above the compose
    /// box) is currently expanded.
    model_picker_open: bool,

    /// Files the agent created/edited this turn, for the review bar (Undo all /
    /// Keep) — each carries its captured pre-edit original so Undo can restore it.
    pending_edits: Vec<PendingEdit>,
    review_expanded: bool,
    /// Start indices of the collapsed tool-step runs the user has opened.
    expanded_step_groups: HashSet<usize>,
    /// Whether the saved-chat list is showing.
    chat_picker_open: bool,
    selected_review_path: Option<String>,
    active_side_tool: GenesiSideTool,
    project_canvas_state: ProjectCanvasState,
    project_canvas_generation: u64,
    selected_canvas_node: Option<String>,
    project_canvas_positions: Arc<HashMap<String, Vector2F>>,
    project_canvas_pan: Vector2F,
    project_canvas_zoom: f32,
    project_canvas_drag: Option<ProjectCanvasDragState>,
    project_canvas_last_drag_frame: Option<Instant>,
    project_canvas_last_drag_pointer: Option<Vector2F>,
    project_canvas_pending_drag_pointer: Option<Vector2F>,
    project_canvas_palette_scroll: ClippedScrollStateHandle,
    project_canvas_inspector_scroll: ClippedScrollStateHandle,
    /// Size of the graph surface, measured by the canvas element during layout
    /// and read back here so off-screen nodes are never even built.
    pub(super) project_canvas_viewport: CanvasViewport,

    // ── Genesi Probe ──
    probe_state: ProbeState,
    /// Bumped on every survey so a slow scan that finishes after the user moved
    /// to another project cannot overwrite the newer result.
    probe_generation: u64,
    /// Same guard for in-flight requests: pressing Send twice must not let the
    /// first response land on top of the second.
    probe_request_generation: u64,
    probe_method: String,
    probe_url: ViewHandle<EditorView>,
    probe_headers_input: ViewHandle<EditorView>,
    probe_body_input: ViewHandle<EditorView>,
    probe_selected_route: Option<String>,
    /// Exactly what Probe last wrote into the URL field. Lets a survey that
    /// finishes later tell its own seed apart from a URL the user typed, so it
    /// can correct the port without ever clobbering an edit.
    probe_seeded_url: Option<String>,
    /// A Canvas endpoint clicked before the survey knew the port: method,
    /// route, body hint. Consumed by `apply_probe_survey`.
    probe_pending_route: Option<(String, String, Option<String>)>,
    probe_response: ProbeResponseState,
    probe_history: Vec<ProbeHistoryEntry>,
    /// Which pane the request editor shows: headers when set, body otherwise.
    probe_show_request_headers: bool,
    probe_show_response_headers: bool,
    /// The reading lenses for the focused file, recomputed when it changes.
    lenses: Option<(ArchitectureLens, PerformanceLens)>,
    lenses_generation: u64,
    /// `true` shows the performance lens, `false` architecture.
    lens_performance: bool,
    lenses_scroll: ClippedScrollStateHandle,
    probe_sidebar_scroll: ClippedScrollStateHandle,
    probe_response_scroll: ClippedScrollStateHandle,

    soundscape_enabled: bool,
    soundscape_index: usize,
    keyboard_asmr_enabled: bool,
    keyboard_switch_index: usize,
}

impl LocalAiChatView {
    pub fn new(ctx: &mut ViewContext<Self>) -> Self {
        let input = ctx.add_typed_action_view(|ctx| SubmittableTextInput::new(ctx));
        input.update(ctx, |input, ctx| {
            input.set_placeholder_text(" Ask the local model...", ctx);
            input.set_outer_margins(0., 0., ctx);
            input.set_submit_button_visible(false, ctx);
            // The compose box already IS the surface; the component's own frame
            // drew a second box inside it.
            input.set_borderless(true, ctx);
            // Own paste: a file copied in the file manager becomes an attachment
            // chip, not a wall of path text (see `handle_input_event`).
            input.set_delegate_paste(true, ctx);
        });
        ctx.subscribe_to_view(&input, |me, _, event, ctx| {
            me.handle_input_event(event, ctx);
        });

        // Probe's three fields are plain editors rather than
        // `SubmittableTextInput`s: that component clears its buffer on submit,
        // which is right for a chat compose box and wrong for a URL you want to
        // tweak and send again.
        let probe_url = Self::probe_editor(
            ctx,
            true,
            "http://127.0.0.1:3000/api/users  ·  or just localhost:8080/health",
        );
        ctx.subscribe_to_view(&probe_url, |me, _, event, ctx| {
            // Enter in the URL field sends, the way every API client behaves.
            if matches!(event, EditorEvent::Enter) {
                me.send_probe_request(ctx);
            }
        });
        let probe_headers_input = Self::probe_editor(
            ctx,
            false,
            "Authorization: Bearer …\nAccept: application/json",
        );
        let probe_body_input = Self::probe_editor(ctx, false, "{\n  \"name\": \"value\"\n}");

        let mut view = Self {
            weak_handle: ctx.handle(),
            input,
            messages: Vec::new(),
            active_chat_id: String::new(),
            chats: Vec::new(),
            transcript_scroll: ClippedScrollStateHandle::default(),
            transcript_selection: SelectionHandle::default(),
            selected_transcript_text: Arc::new(RwLock::new(None)),
            review_sidebar_scroll: ClippedScrollStateHandle::default(),
            vibe_mode: false,
            endpoint: LocalEndpoint::Ollama,
            turbo_available: false,
            endpoint_user_chosen: false,
            models: Vec::new(),
            gguf_labels: HashMap::new(),
            preparing_model: None,
            mode_picker_open: false,
            picker_tab: ModelPickerTab::Local,
            selected_model: None,
            ai_mode: None,
            in_flight: false,
            steering: Vec::new(),
            error: None,
            current_turn: 0,
            attach_context: true,
            attachments: Vec::new(),
            // Chat, not Build. Build runs an agent loop, and a loop is many
            // requests for one prompt -- up to MAX_AGENT_STEPS of them. That is
            // the right shape when you asked for work to be done, and the wrong
            // one to charge someone's metered API key for by default, when the
            // prompt was a question. Build stays one click away, and the choice
            // is remembered once made.
            agent_mode: false,
            agent_root: None,
            agent_messages: Vec::new(),
            agent_model: String::new(),
            agent_step: 0,
            agent_step_buffer: String::new(),
            agent_reasoning_buffer: String::new(),
            agent_native_call: None,
            agent_nudges: 0,
            native_tools: NativeToolSupport::Untried,
            server_context: None,
            agent_tool_summary: String::new(),
            auto_approve: false,
            pending_tool: None,
            cloud: load_cloud_config().unwrap_or_default().with_defaults(),
            cloud_keys: CloudKeyStore::default(),
            cloud_active: false,
            input_mode: InputMode::Chat,
            model_picker_open: false,
            pending_edits: Vec::new(),
            review_expanded: false,
            expanded_step_groups: HashSet::new(),
            chat_picker_open: false,
            selected_review_path: None,
            active_side_tool: GenesiSideTool::Review,
            project_canvas_state: ProjectCanvasState::NoProject,
            project_canvas_generation: 0,
            selected_canvas_node: None,
            project_canvas_positions: Arc::new(HashMap::new()),
            project_canvas_pan: vec2f(42., 42.),
            project_canvas_zoom: 1.,
            project_canvas_drag: None,
            project_canvas_last_drag_frame: None,
            project_canvas_last_drag_pointer: None,
            project_canvas_pending_drag_pointer: None,
            project_canvas_palette_scroll: ClippedScrollStateHandle::default(),
            project_canvas_inspector_scroll: ClippedScrollStateHandle::default(),
            project_canvas_viewport: CanvasViewport::default(),
            probe_state: ProbeState::NoProject,
            probe_generation: 0,
            probe_request_generation: 0,
            probe_method: "GET".to_string(),
            probe_url,
            probe_headers_input,
            probe_body_input,
            probe_selected_route: None,
            probe_seeded_url: None,
            probe_pending_route: None,
            probe_response: ProbeResponseState::Idle,
            probe_history: Vec::new(),
            probe_show_request_headers: false,
            probe_show_response_headers: false,
            lenses: None,
            lenses_generation: 0,
            lens_performance: false,
            lenses_scroll: ClippedScrollStateHandle::default(),
            probe_sidebar_scroll: ClippedScrollStateHandle::default(),
            probe_response_scroll: ClippedScrollStateHandle::default(),
            soundscape_enabled: false,
            soundscape_index: 0,
            keyboard_asmr_enabled: false,
            keyboard_switch_index: 0,
        };
        view.load_cloud_keys(ctx);
        view.load_mempalace(ctx);
        view.refresh_ai_mode();
        view.refresh_models(ctx);
        view
    }

    /// Name of the currently selected model, if any. In cloud mode this is the
    /// configured provider model; otherwise the picked local model.
    fn current_model(&self) -> Option<String> {
        if self.cloud_active {
            let model = self.cloud.model.trim();
            return (!model.is_empty()).then(|| model.to_string());
        }
        self.selected_model
            .and_then(|index| self.models.get(index))
            .cloned()
    }

    /// Friendly name for a model reference. Ollama tags read fine as-is; a
    /// `gguf:` reference becomes the model's real name so the picker doesn't
    /// show raw file stems.
    fn model_label(&self, reference: &str) -> String {
        self.gguf_labels
            .get(reference)
            .cloned()
            .unwrap_or_else(|| reference.to_string())
    }

    /// A local GGUF is loaded by llama-server, never by ollama, so picking one
    /// has to (a) switch the endpoint to Turbo and (b) make that server actually
    /// open THIS file. Without (b) the request goes to a server holding a
    /// different model — or none — and returns nothing, which is precisely the
    /// "runs a few seconds and answers nothing" failure.
    fn ensure_model_ready(&mut self, model: String, ctx: &mut ViewContext<Self>) {
        if self.cloud_active {
            return;
        }
        // A GGUF can only be loaded by llama-server, so choosing one chooses Turbo.
        if is_gguf_ref(&model) && self.endpoint != LocalEndpoint::Turbo {
            self.endpoint = LocalEndpoint::Turbo;
            self.endpoint_user_chosen = true;
        }
        // Ollama serves its own tags with no help. Only Turbo has to be TOLD what
        // to load — and it does have to be told for a plain tag too, or it keeps
        // serving whatever model it happened to have open.
        if self.endpoint != LocalEndpoint::Turbo {
            return;
        }
        if self.preparing_model.as_deref() == Some(model.as_str()) {
            return; // already bringing this one up
        }
        self.preparing_model = Some(model.clone());
        self.error = None;
        ctx.notify();
        let label = self.model_label(&model);
        ctx.spawn(
            {
                let model = model.clone();
                async move { ensure_turbo_serving(&model).await }
            },
            move |me, result, ctx| {
                if me.preparing_model.as_deref() != Some(model.as_str()) {
                    return; // superseded by another pick
                }
                me.preparing_model = None;
                match result {
                    Ok(()) => {
                        me.turbo_available = true;
                        me.error = None;
                    }
                    Err(err) => {
                        me.error = Some(format!("Couldn't load {label}: {err}"));
                    }
                }
                ctx.notify();
            },
        );
    }

    /// Build the chat stream for the active backend: the user's cloud provider
    /// (BYOK) when it's selected and ready, otherwise the local Ollama / Turbo
    /// endpoint. Both yield the same `ChatStreamItem` stream so the agent loop
    /// and plain chat don't care which one is in use.
    fn build_stream(
        &mut self,
        model: &str,
        messages: Vec<ChatMessage>,
        ctx: &mut ViewContext<Self>,
    ) -> futures::stream::BoxStream<'static, Result<ChatStreamItem>> {
        self.load_cloud_keys(ctx);
        // Hand a tool-native model a real schema when we're driving the agent and
        // the server is known to take one. gpt-oss reaches for its own tool
        // channel however the prompt is worded, so the text protocol alone leaves
        // it narrating ("I'll use list_files") instead of acting.
        let tools = (self.agent_mode && self.native_tools != NativeToolSupport::Unsupported)
            .then(local_agent::tool_schemas);
        // Fit the conversation to the window BEFORE asking for a reply, and size
        // the reply against what's left. Sending a fixed max_tokens was its own
        // bug: asking for 4096 reply tokens on a 4096-token window leaves the
        // prompt nowhere to go.
        // llama-server renders the tool schemas INTO the prompt, so they cost
        // real context — but they travel in their own request field and were
        // invisible to the budget below, which only ever weighed message text.
        // On an 8K window the six schemas plus their descriptions were enough to
        // push a Build turn past the wall, and llama-server answers a request
        // that overflows by accepting it and streaming nothing at all. Chat mode
        // sends no tools, which is exactly why only Build broke.
        let tool_tokens = tools
            .as_ref()
            .map(|schemas| estimate_tokens(&schemas.to_string()))
            .unwrap_or(0);
        let messages = self.fit_to_context(messages, tool_tokens);
        let prompt_tokens = messages
            .iter()
            .map(|message| estimate_tokens(&message.content))
            .sum::<usize>()
            + tool_tokens;
        // Only the local server needs a reply cap; the cloud path sends none.
        let reply_tokens = reply_budget(self.effective_context(), prompt_tokens);
        if self.cloud_active && self.cloud_ready() {
            stream_chat_cloud(
                self.cloud.provider,
                model,
                self.active_cloud_key().to_string(),
                messages,
                tools,
            )
        } else {
            // The transport is decided by the MODEL, not by `self.endpoint`
            // alone — see transport_for.
            stream_chat(
                transport_for(model, self.endpoint),
                model,
                messages,
                tools,
                reply_tokens,
            )
        }
    }

    /// Shrink a request until it fits the server's context window.
    ///
    /// The agent loop appends every tool result to the conversation and never
    /// dropped anything, while a single `read_file` can carry 24 KB — about 6k
    /// tokens. On the 4K window llama-server actually runs with, the first file
    /// the model read blew the budget and the turn died mid-task with "request
    /// (4119 tokens) exceeds the available context size (4096 tokens)". That read
    /// as "the AI randomly stops", because nothing in the UI connected the two.
    ///
    /// Order of sacrifice, least useful first: the OLDEST exchanges go before the
    /// newest (the model needs where it is, not where it started), and only then
    /// do we start cutting into the bodies of what is left.
    /// `reserved` is context the request spends outside the messages — today the
    /// tool schemas, which the server folds into the prompt for us.
    fn fit_to_context(&self, messages: Vec<ChatMessage>, reserved: usize) -> Vec<ChatMessage> {
        // A cloud provider's window is orders of magnitude bigger and unknown to
        // us; trimming to a local-sized budget would throw away context for no
        // reason.
        if self.cloud_active {
            return messages;
        }
        let n_ctx = self.server_context.unwrap_or(ASSUMED_CONTEXT_TOKENS) as usize;
        // Leave the model room to actually answer.
        let budget = n_ctx
            .saturating_sub(REPLY_RESERVE_TOKENS)
            .saturating_sub(reserved)
            .max(512);

        let cost = |m: &ChatMessage| estimate_tokens(&m.content);
        let total = |ms: &[ChatMessage]| -> usize { ms.iter().map(cost).sum() };
        if total(&messages) <= budget {
            return messages;
        }

        // The leading system messages are the instructions plus the file context
        // the user deliberately attached; they stay unless nothing else is left.
        let head_len = messages.iter().take_while(|m| m.role == "system").count();
        let mut head: Vec<ChatMessage> = messages[..head_len].to_vec();
        let mut tail: Vec<ChatMessage> = messages[head_len..].to_vec();

        // 1. Drop the oldest turns, always keeping the most recent one.
        while total(&head) + total(&tail) > budget && tail.len() > 1 {
            tail.remove(0);
        }

        // 2. Still over: cut into bodies, biggest first, so one huge tool result
        //    can't starve everything else. One flat list keeps the "which is
        //    biggest" question simple; bounded so it always terminates.
        head.extend(tail);
        let mut messages = head;
        for _ in 0..64 {
            let over = total(&messages).saturating_sub(budget);
            if over == 0 {
                break;
            }
            // NEVER cut the agent's instructions. They are what tells the model
            // that run_command and the other tools exist at all, so trimming
            // them turns the agent back into a chatbot that says it "has no
            // direct access to run commands" — silently, with nothing in the UI
            // to explain why. Tool results and old turns are expendable; this is
            // not.
            let Some(index) = (0..messages.len())
                .filter(|i| !(i == &0 && messages[0].role == "system"))
                .max_by_key(|i| cost(&messages[*i]))
            else {
                break;
            };
            let current = cost(&messages[index]);
            if current <= MIN_KEPT_MESSAGE_TOKENS {
                break; // everything is already at the floor
            }
            let target = current.saturating_sub(over).max(MIN_KEPT_MESSAGE_TOKENS);
            messages[index].content = clamp_to_tokens(&messages[index].content, target);
        }
        messages
    }

    /// The context window to budget against, and whether it is a real reading.
    fn effective_context(&self) -> u32 {
        self.server_context.unwrap_or(ASSUMED_CONTEXT_TOKENS)
    }

    /// Re-read the daemon's published AI Mode state (cheap, synchronous).
    fn refresh_ai_mode(&mut self) {
        self.ai_mode = read_ai_mode_state();
    }

    /// Ask the active endpoint for its model list and probe whether Turbo is up.
    fn refresh_models(&mut self, ctx: &mut ViewContext<Self>) {
        // In cloud (BYOK) mode there's no local server to enumerate — the model is
        // whatever the user configured, so skip the local probes entirely.
        if self.cloud_active {
            return;
        }
        // ALWAYS enumerate ollama, whichever endpoint is selected. Asking Turbo
        // instead only ever returns the one file llama-server has open (which is
        // then filtered out below), so on Turbo the list collapsed to the GGUF
        // library alone: a model fetched with `ollama pull` vanished, and the
        // selection re-anchored to the first GGUF — which is why picking Turbo
        // kept showing gpt-oss no matter what was actually loaded. Turbo can
        // serve an ollama tag perfectly well (`genesi-ai-turbo serve <tag>`
        // resolves it to its blob), so both endpoints offer the same list.
        let base = DEFAULT_LOCAL_BASE_URL.to_string();
        // Turbo (llama-server) already has its model loaded and its `/v1/models`
        // isn't a reliable signal, so an empty/failed list there is not an error —
        // readiness comes from the `/health` probe below instead.
        let is_turbo = self.endpoint == LocalEndpoint::Turbo;
        ctx.spawn(
            async move {
                // Two sources, one list. `/v1/models` only knows what ollama
                // pulled, so a GGUF the user imported into Genesi is invisible to
                // it — ask genesi-ai-turbo for the library too, exactly like the
                // AI Mode Monitor does, so both apps offer the same models.
                let served = list_models(&base).await;
                let local = list_gguf_models().await;
                (served, local)
            },
            move |me, (served, local), ctx| {
                let reachable = served.is_ok();
                let mut models = served.unwrap_or_default();
                // Drop llama-server's self-report. On :11435 `/v1/models` answers
                // with the FILE it currently has open — a .gguf path, or an
                // extensionless ollama blob like
                // `/var/lib/ollama/blobs/sha256-7da77af9f7ccdff`. Neither is a
                // model you can choose: they only describe what Turbo already
                // loaded, and listing them put that hash in the picker as if it
                // were a model name. The real choices are the ollama tags and the
                // GGUF library, both added below.
                models.retain(|m| !is_gguf_ref(m) && !m.contains('/'));
                me.gguf_labels = local.iter().cloned().collect();
                models.extend(local.into_iter().map(|(reference, _)| reference));

                // Re-anchor the selection by VALUE, not index. The list is rebuilt
                // from two async sources, so a plain index survives a reorder by
                // silently pointing at a different model.
                let previous = me.current_model();
                me.models = models;
                me.selected_model = if me.models.is_empty() {
                    None
                } else {
                    previous
                        .and_then(|model| me.models.iter().position(|m| *m == model))
                        .or(Some(0))
                };
                me.error = if !me.models.is_empty() || is_turbo {
                    None
                } else if reachable {
                    Some(
                        "No local models found. Is ollama running? Try \
                         `ollama pull llama3.2`, or import a .gguf in the AI Mode \
                         Monitor."
                            .to_string(),
                    )
                } else {
                    Some("Can't reach the local endpoint (is ollama running?)".to_string())
                };
                ctx.notify();
            },
        );

        // Probe liveness and the REAL context window together: `serve` reuses an
        // already-running server, so the window Code asked for is routinely not
        // the window it gets.
        ctx.spawn(
            async {
                let available = turbo_health_ok().await;
                let context = if available {
                    turbo_context_size().await
                } else {
                    None
                };
                (available, context)
            },
            |me, (available, context), ctx| {
                if let Some(context) = context {
                    me.server_context = Some(context);
                }
                me.turbo_available = available;
                // Default to the shared Turbo daemon when it's up and the user
                // hasn't deliberately picked an endpoint — Code then inherits
                // GPU offload, the q8 KV cache and the warm model for free
                // (roadmap 4.1 / 4.0). One-shot: switching sets the endpoint, so
                // the re-probe below sees `== Turbo` and won't loop.
                if available && !me.endpoint_user_chosen && me.endpoint != LocalEndpoint::Turbo {
                    me.endpoint = LocalEndpoint::Turbo;
                    me.refresh_models(ctx);
                }
                ctx.notify();
            },
        );
    }

    fn scroll_to_bottom(&self) {
        self.transcript_scroll
            .scroll_to(Pixels::new(SCROLL_TO_BOTTOM));
    }

    fn handle_input_event(
        &mut self,
        event: &SubmittableTextInputEvent,
        ctx: &mut ViewContext<Self>,
    ) {
        match event {
            // In a cloud key/model entry mode the submit is the value to save;
            // otherwise it's a chat prompt. Route the prompt through the workspace
            // so it can attach the focused file as context before we send (see
            // `LocalAiChatEvent::SubmitPrompt`).
            SubmittableTextInputEvent::Submit(text) => match self.input_mode {
                InputMode::CloudKey | InputMode::CloudModel => {
                    self.save_cloud_field(self.input_mode, text.clone(), ctx);
                }
                InputMode::Chat => ctx.emit(LocalAiChatEvent::SubmitPrompt(text.clone())),
            },
            // Escape backs out of a key/model entry without saving.
            SubmittableTextInputEvent::Escape => {
                if self.input_mode != InputMode::Chat {
                    self.set_input_mode(InputMode::Chat, ctx);
                }
            }
            // We took paste over from the editor, so we owe it the text case.
            // Files and images become attachment chips; anything else is typed
            // in as usual. Key/model entry never attaches — a pasted API key is
            // text, always.
            // Ctrl+C with nothing selected in the compose box: the user had
            // selected a message in the TRANSCRIPT, whose selection the editor
            // knows nothing about. It used to swallow the key and copy nothing.
            SubmittableTextInputEvent::Copy => {
                if let Some(text) = self
                    .selected_transcript_text
                    .read()
                    .clone()
                    .filter(|text| !text.is_empty())
                {
                    ctx.clipboard().write(ClipboardContent::plain_text(text));
                }
            }
            SubmittableTextInputEvent::Paste => {
                if self.input_mode == InputMode::Chat && self.attach_from_clipboard(ctx) {
                    return;
                }
                let text = ctx.clipboard().read().plain_text;
                if !text.is_empty() {
                    self.input
                        .update(ctx, |input, ctx| input.insert_pasted_text(&text, ctx));
                }
            }
        }
    }

    /// A friendly name for the configured cloud provider (or "cloud").
    fn cloud_label(&self) -> String {
        self.cloud.provider.label().to_string()
    }

    fn active_cloud_key(&self) -> &str {
        self.cloud_keys.get(self.cloud.provider)
    }

    fn cloud_ready(&self) -> bool {
        self.cloud.is_ready(self.active_cloud_key())
    }

    fn active_backend_error_label(&self) -> String {
        if self.cloud_active {
            format!("{} error", self.cloud.provider.label())
        } else {
            "Local model error".to_string()
        }
    }

    fn save_cloud_keys(&self, ctx: &mut ViewContext<Self>) -> Result<()> {
        let json = serde_json::to_string(&self.cloud_keys)?;
        ctx.secure_storage()
            .write_value_with_owner_only_fallback(CLOUD_KEYS_STORAGE_KEY, &json)
            .map_err(|e| anyhow::anyhow!("failed to write cloud keys: {e}"))
    }

    fn load_cloud_keys(&mut self, ctx: &mut ViewContext<Self>) {
        match ctx.secure_storage().read_value(CLOUD_KEYS_STORAGE_KEY) {
            Ok(json) => match serde_json::from_str::<CloudKeyStore>(&json) {
                Ok(keys) => self.cloud_keys = keys,
                Err(err) => {
                    log::warn!("Failed to deserialize cloud keys from secure storage: {err:#}");
                }
            },
            Err(err) => {
                if !matches!(err, warpui_extras::secure_storage::Error::NotFound) {
                    log::warn!("Failed to read cloud keys from secure storage: {err:#}");
                }
            }
        }

        if let Some((provider, key)) = load_legacy_cloud_key() {
            let existing = self.cloud_keys.get(provider).trim();
            if existing.is_empty() && !key.trim().is_empty() {
                self.cloud_keys.set(provider, key);
                if let Err(err) = self.save_cloud_keys(ctx) {
                    log::warn!("Failed to migrate legacy cloud key to secure storage: {err:#}");
                }
                let _ = save_cloud_config(&self.cloud);
            }
        }
    }

    fn select_cloud_provider(
        &mut self,
        provider: CloudProviderKind,
        close_picker: bool,
        ctx: &mut ViewContext<Self>,
    ) {
        self.load_cloud_keys(ctx);
        self.endpoint_user_chosen = true;
        self.cloud_active = true;
        self.endpoint = LocalEndpoint::Ollama;
        let old_provider = self.cloud.provider;
        let old_default = old_provider.default_model();
        self.cloud.provider = provider;
        if self.cloud.model.trim().is_empty() || self.cloud.model == old_default {
            self.cloud.model = provider.default_model().to_string();
        }
        if close_picker {
            self.model_picker_open = false;
        }
        if let Err(err) = save_cloud_config(&self.cloud) {
            self.error = Some(format!("Couldn't save cloud config: {err}"));
        } else if !self.cloud_ready() {
            self.error = Some(format!(
                "Add your {} API key in Settings to use {}.",
                self.cloud.provider.label(),
                self.cloud.model
            ));
        } else {
            self.error = None;
        }
    }

    /// Switch what the compose box captures next, updating its placeholder.
    fn set_input_mode(&mut self, mode: InputMode, ctx: &mut ViewContext<Self>) {
        self.input_mode = mode;
        let placeholder = match mode {
            InputMode::Chat => " Ask the model...".to_string(),
            InputMode::CloudKey => {
                format!(" Paste your {} API key, then Enter", self.cloud_label())
            }
            InputMode::CloudModel => {
                format!(
                    " Model id for {} (e.g. {}), then Enter",
                    self.cloud_label(),
                    {
                        let m = self.cloud.model.trim();
                        if m.is_empty() {
                            self.cloud.provider.default_model()
                        } else {
                            m
                        }
                    }
                )
            }
        };
        self.input.update(ctx, |input, ctx| {
            input.set_placeholder_text(placeholder, ctx)
        });
        ctx.notify();
    }

    /// Persist a typed key or model id onto the cloud config, then return to chat.
    fn save_cloud_field(&mut self, which: InputMode, value: String, ctx: &mut ViewContext<Self>) {
        let value = value.trim().to_string();
        match which {
            InputMode::CloudKey => self.cloud_keys.set(self.cloud.provider, value),
            InputMode::CloudModel => self.cloud.model = value,
            InputMode::Chat => {}
        }
        let save_result = match which {
            InputMode::CloudKey => self.save_cloud_keys(ctx),
            InputMode::CloudModel => save_cloud_config(&self.cloud),
            InputMode::Chat => Ok(()),
        };
        match save_result {
            Ok(()) => self.error = None,
            Err(e) => self.error = Some(format!("Couldn't save cloud config: {e}")),
        }
        self.set_input_mode(InputMode::Chat, ctx);
    }

    // ─────────────────── Steering: correcting a running turn ───────────────────

    /// Take a correction for the turn already in progress.
    ///
    /// It goes into the transcript immediately so the user can see it landed —
    /// the alternative is typing into a box that appears to swallow input, which
    /// is exactly what happened before steering existed.
    fn steer(&mut self, text: String, ctx: &mut ViewContext<Self>) {
        self.messages.push(ChatEntry::prose(
            ChatRole::User,
            text.clone(),
            Some("steering".to_string()),
        ));
        self.steering.push(text);
        self.scroll_to_bottom();
        self.persist_mempalace(ctx, false);
        ctx.notify();
    }

    /// Fold queued corrections into the running agent conversation.
    ///
    /// Called only from the top of [`Self::run_agent_step`], where the previous
    /// stream has ended and the next has not been built — the one point where a
    /// running turn can change course without touching the streaming path at
    /// all. They are labelled as mid-task corrections because a model that reads
    /// them as ordinary follow-ups tends to finish the original plan first and
    /// address them afterwards, which is the opposite of steering.
    fn apply_steering_to_agent(&mut self) {
        if self.steering.is_empty() {
            return;
        }
        for correction in std::mem::take(&mut self.steering) {
            self.agent_messages.push(ChatMessage::user(format!(
                "Course correction from the user, mid-task. Apply this to what you \
                 are doing now, before continuing:\n\n{correction}"
            )));
        }
        // The correction is fresh instruction, so the model has something to act
        // on again — don't hold last step's "it only thought" count against it.
        self.agent_nudges = 0;
    }

    /// Anything still queued once a turn settles becomes the next prompt.
    ///
    /// Plain chat has a single stream and so no step boundary to inject at, and
    /// an agent turn can finish before it ever takes another step. Either way
    /// the user typed something and is owed an answer to it.
    fn flush_pending_steering(&mut self, ctx: &mut ViewContext<Self>) {
        if self.steering.is_empty() || self.in_flight {
            return;
        }
        let prompt = std::mem::take(&mut self.steering).join("\n\n");
        // Back out through the workspace so the follow-up picks up editor
        // context exactly the way a typed prompt would.
        ctx.emit(LocalAiChatEvent::SubmitPrompt(prompt));
    }

    pub fn cancel_steering(&mut self, ctx: &mut ViewContext<Self>) {
        if self.steering.is_empty() {
            return;
        }
        self.steering.clear();
        ctx.notify();
    }

    /// Send the prompt and start streaming the assistant's reply, optionally
    /// attaching the focused file as context. Called by the workspace after it
    /// gathers `context` for the current turn.
    pub fn send_with_context(
        &mut self,
        prompt: String,
        context: Option<CodeContext>,
        project_root: Option<PathBuf>,
        ctx: &mut ViewContext<Self>,
    ) {
        let prompt = prompt.trim().to_string();
        if prompt.is_empty() {
            return;
        }
        // Mid-flight steering. This used to `return` and lose the text
        // entirely; now it joins the turn already in progress.
        if self.in_flight {
            self.steer(prompt, ctx);
            return;
        }
        self.load_cloud_keys(ctx);
        // BYOK: a selected cloud provider needs its key before it can be used.
        if self.cloud_active && !self.cloud_ready() {
            self.error = Some(format!(
                "Add your {} API key in Genesi AI settings first.",
                self.cloud.provider.label()
            ));
            ctx.notify();
            return;
        }
        // Turbo serves whatever model llama-server already loaded, so the `model`
        // field is informational there — fall back to a placeholder when no model
        // is listed. Ollama and cloud providers need a real model name.
        let model = match self.current_model() {
            Some(model) => model,
            None if self.cloud_active => {
                self.error =
                    Some("Set a model for the cloud provider (the Model chip).".to_string());
                ctx.notify();
                return;
            }
            None if self.endpoint == LocalEndpoint::Turbo => "local".to_string(),
            None => {
                self.error = Some(
                    "No model selected. Is ollama running? Try `ollama pull llama3.2`, \
                     or import a .gguf in the AI Mode Monitor."
                        .to_string(),
                );
                ctx.notify();
                return;
            }
        };

        // A GGUF still being loaded into llama-server would take the request and
        // answer nothing. Say so instead of sending into a server that isn't ready.
        // Only THIS model blocks: a load left running for a model the user has
        // since switched away from must not hold the send button hostage.
        if self.preparing_model.as_deref() == Some(model.as_str()) {
            self.error = Some(format!(
                "{} is still loading — give it a moment.",
                self.model_label(&model)
            ));
            ctx.notify();
            return;
        }
        // Headed for Turbo but the server isn't up (Code restarted, or Turbo was
        // stopped elsewhere)? Bring it back rather than failing the turn. Keyed on
        // the TRANSPORT, not on the model being a GGUF: an ollama tag served
        // through Turbo needs the server just as much.
        if transport_for(&model, self.endpoint) == LocalEndpoint::Turbo && !self.turbo_available {
            self.ensure_model_ready(model.clone(), ctx);
            self.error = Some(format!(
                "Loading {} — press send again once it's ready.",
                self.model_label(&model)
            ));
            ctx.notify();
            return;
        }

        self.error = None;

        // Attach the focused file only when the toggle is on. Recorded on the
        // user entry so the transcript shows what the model was given.
        let context = if self.attach_context { context } else { None };
        // The turn consumes whatever the user pinned: the chips clear, and the
        // transcript keeps the record of what went with the message.
        let attachments = std::mem::take(&mut self.attachments);
        let context_label = {
            let mut parts: Vec<String> = context
                .as_ref()
                .map(CodeContext::label)
                .into_iter()
                .collect();
            parts.extend(attachments.iter().map(|a| a.name.clone()));
            (!parts.is_empty()).then(|| parts.join(" · "))
        };

        self.messages
            .push(ChatEntry::prose(ChatRole::User, prompt, context_label));
        self.persist_mempalace(ctx, true);

        // System prompt: the agent variant (with tool instructions) in agent
        // mode, otherwise the plain chat prompt. Then the attached file context,
        // then the visible transcript text (tool steps are UI-only).
        let system_prompt = if self.agent_mode {
            local_agent::agent_system_prompt()
        } else {
            SYSTEM_PROMPT.to_string()
        };
        let mut request = vec![ChatMessage::system(system_prompt)];
        if let Some(context) = &context {
            request.push(context.to_system_message());
        }
        // Files the user pinned. Text goes in as fenced system context; images
        // ride on the user's own message, which is the only place the OpenAI and
        // Anthropic shapes both accept them.
        let can_see = model_supports_vision(&model);
        let mut images = Vec::new();
        for attachment in &attachments {
            match attachment.kind {
                AttachmentKind::Text => {
                    if let Some(message) = attachment.to_system_message() {
                        request.push(message);
                    }
                }
                AttachmentKind::Image => {
                    match can_see.then(|| attachment.as_chat_image()).flatten() {
                        Some(image) => images.push(image),
                        // Silently dropping it would look like the model ignored the
                        // picture. Tell the model it exists so it can say so.
                        None => request.push(ChatMessage::system(format!(
                            "The user attached the image `{}`, but this model cannot read \
                         images, so its contents are not available. Say so if it matters.",
                            attachment.name
                        ))),
                    }
                }
            }
        }
        let last_user_index = self
            .messages
            .iter()
            .rposition(|entry| entry.role == ChatRole::User && !entry.text.is_empty());
        for (index, entry) in self.messages.iter().enumerate() {
            if entry.text.is_empty() {
                continue;
            }
            match entry.role {
                ChatRole::User => {
                    let mut message = ChatMessage::user(entry.text.clone());
                    // Only the turn being sent carries the images.
                    if Some(index) == last_user_index && !images.is_empty() {
                        message = message.with_images(std::mem::take(&mut images));
                    }
                    request.push(message);
                }
                ChatRole::Assistant => request.push(ChatMessage::assistant(entry.text.clone())),
                // Thought / Tool / Command steps are UI-only; the model's real
                // tool calls + results live in `agent_messages` during a turn.
                ChatRole::Thought | ChatRole::Tool | ChatRole::Command => {}
            }
        }

        // A fresh generation: bump the turn id so any still-running callbacks
        // from a previous (e.g. just-stopped) turn are ignored.
        self.current_turn += 1;
        self.in_flight = true;
        self.scroll_to_bottom();

        if self.agent_mode {
            // Drive the agent loop: the model may call read tools before it
            // answers. Each step appends to `agent_messages`.
            self.agent_root = project_root;
            self.agent_model = model;
            self.agent_messages = request;
            self.agent_step = 0;
            self.agent_nudges = 0;
            self.run_agent_step(ctx);
            ctx.notify();
            return;
        }

        // Plain chat: one streamed reply into a placeholder bubble.
        self.messages
            .push(ChatEntry::prose(ChatRole::Assistant, String::new(), None));
        let turn = self.current_turn;
        let stream = self.build_stream(&model, request, ctx);
        ctx.spawn_stream_local(
            stream,
            move |me, item, ctx| me.on_stream_item(turn, item, ctx),
            move |me, ctx| {
                // Stream ended without an explicit `[DONE]` — settle the UI.
                if turn == me.current_turn && me.in_flight {
                    me.in_flight = false;
                    // An END with nothing streamed is a FAILURE, not a finished
                    // answer. Settling silently here is what produced the
                    // "spins for a few seconds, then nothing" report: the empty
                    // bubble stayed and no reason was ever shown. Say something.
                    let empty = me
                        .messages
                        .last()
                        .is_some_and(|m| m.role == ChatRole::Assistant && m.text.is_empty());
                    if empty {
                        me.messages.pop();
                        me.error = Some(me.empty_reply_error());
                    }
                    me.flush_pending_steering(ctx);
                    ctx.notify();
                }
            },
        );
        ctx.notify();
    }

    // ── attachments ────────────────────────────────────────────────────────

    /// Open the OS file picker and pin whatever the user chooses to the next
    /// message. Multi-select, no type filter: an attachment is a reference, and
    /// what counts as a useful reference is the user's call, not ours.
    fn pick_attachments(&mut self, ctx: &mut ViewContext<Self>) {
        ctx.open_file_picker(
            move |result, ctx| match result {
                Ok(paths) => ctx.dispatch_typed_action(&LocalAiChatAction::AttachPaths(
                    paths.into_iter().map(PathBuf::from).collect(),
                )),
                Err(err) => log::warn!("attachment picker failed: {err}"),
            },
            FilePickerConfiguration::new().allow_multi_select(),
        );
    }

    /// Pin file paths, skipping ones already attached so a double paste doesn't
    /// send the same file twice.
    fn attach_paths(&mut self, paths: impl Iterator<Item = PathBuf>, ctx: &mut ViewContext<Self>) {
        let mut added = false;
        for path in paths {
            if path.as_os_str().is_empty() || path.is_dir() {
                continue;
            }
            if self
                .attachments
                .iter()
                .any(|existing| existing.path.as_deref() == Some(path.as_path()))
            {
                continue;
            }
            self.attachments.push(ChatAttachment::from_path(path));
            added = true;
        }
        if added {
            self.error = None;
            ctx.notify();
        }
    }

    /// Attach a file dragged out of the file tree. Public because the drop is
    /// handled where the drag started (the file tree owns the path), and it lands
    /// here through [`LocalAiDropTargetData`].
    pub fn attach_dropped_path(&mut self, path: PathBuf, ctx: &mut ViewContext<Self>) {
        self.attach_paths(std::iter::once(path), ctx);
    }

    /// Handle a paste into the compose box. Returns true when the clipboard held
    /// files or an image, meaning it became attachments and the text must NOT
    /// also be typed into the box (pasting a file should not paste its path).
    fn attach_from_clipboard(&mut self, ctx: &mut ViewContext<Self>) -> bool {
        let content = ctx.clipboard().read();
        let paths: Vec<PathBuf> = content
            .paths
            .unwrap_or_default()
            .into_iter()
            .map(PathBuf::from)
            .filter(|path| path.is_file())
            .collect();
        let images = content.images.unwrap_or_default();
        if paths.is_empty() && images.is_empty() {
            return false;
        }
        let before = self.attachments.len();
        self.attach_paths(paths.into_iter(), ctx);
        for image in images {
            self.attachments.push(ChatAttachment::from_image_bytes(
                image.filename.unwrap_or_default(),
                &image.mime_type,
                image.data,
            ));
        }
        let changed = self.attachments.len() != before;
        if changed {
            self.error = None;
            ctx.notify();
        }
        // Even when everything was a duplicate the paste WAS about files, so
        // swallow it — inserting the raw paths is never what was meant.
        true
    }

    /// Turn the accumulated native tool-call fragments into a tool, if the model
    /// made one and we can map its name onto something we run.
    fn take_native_tool_call(&mut self) -> Option<AgentTool> {
        let (name, arguments) = self.agent_native_call.take()?;
        let name = name?;
        local_agent::tool_from_native_call(&name, &arguments)
    }

    /// Why a local endpoint can accept a request, stream nothing, and close.
    /// Ordered by how often each cause actually bites on a small local setup.
    fn empty_reply_error(&self) -> String {
        if self.cloud_active {
            return format!(
                "{}: the provider closed the stream without sending any text.",
                self.active_backend_error_label()
            );
        }
        match self.endpoint {
            LocalEndpoint::Turbo => format!(
                "{}: the model returned nothing on any channel. The prompt may not fit \
                 the server's context window — start Turbo with a bigger one \
                 (GENESI_TURBO_CTX=16384; the default is 8192), shorten the \
                 conversation, or remove the reference chips above the input. Run \
                 `genesi-ai-turbo serve <model>` in a terminal to see the server's own \
                 error.",
                self.active_backend_error_label()
            ),
            LocalEndpoint::Ollama => format!(
                "{}: the model returned nothing. Check `ollama ps` — the model may \
                 have failed to load, or the prompt may exceed its context window.",
                self.active_backend_error_label()
            ),
        }
    }

    // ── agent loop ─────────────────────────────────────────────────────────

    /// Run one agent step: stream the model's next message into a fresh
    /// "thought" bubble, then settle it in [`Self::on_agent_step_end`] (execute
    /// a tool and loop, or promote the thought into the final answer).
    fn run_agent_step(&mut self, ctx: &mut ViewContext<Self>) {
        self.agent_step_buffer.clear();
        self.agent_reasoning_buffer.clear();
        self.agent_native_call = None;
        self.messages.push(ChatEntry {
            role: ChatRole::Thought,
            text: String::new(),
            context_label: None,
            tool_title: None,
            command: None,
            collapsed: false,
            status: StepStatus::Running,
            diff_stat: None,
            diff_preview: None,
        });
        // The step boundary is the one safe place to change a running turn's
        // course: the previous stream is finished and the next has not been
        // built, so the correction is simply part of the conversation the model
        // is about to read. Nothing about the streaming path changes.
        self.apply_steering_to_agent();
        let turn = self.current_turn;
        let agent_model = self.agent_model.clone();
        let agent_messages = self.agent_messages.clone();
        let stream = self.build_stream(&agent_model, agent_messages, ctx);
        ctx.spawn_stream_local(
            stream,
            move |me, item, ctx| me.on_agent_token(turn, item, ctx),
            move |me, ctx| me.on_agent_step_end(turn, ctx),
        );
    }

    fn on_agent_token(
        &mut self,
        turn: u64,
        item: Result<ChatStreamItem>,
        ctx: &mut ViewContext<Self>,
    ) {
        if turn != self.current_turn {
            return; // a stale turn (the user hit Stop, or started a new send)
        }
        match item {
            Ok(ChatStreamItem::Token(token)) => {
                self.agent_step_buffer.push_str(&token);
                // Show only the human-facing prose as it streams — never the raw
                // `<tool:…>` markup (that becomes a clean tool step once parsed).
                let visible = local_agent::strip_tool_calls(&self.agent_step_buffer);
                if let Some(last) = self.messages.last_mut() {
                    if last.role == ChatRole::Thought {
                        last.text = visible;
                    }
                }
                self.scroll_to_bottom();
                ctx.notify();
            }
            // Thinking is shown live but MUST NOT enter agent_step_buffer: that
            // buffer is what parse_tool_call reads, and a reasoning model's
            // narration there ("I'll create index.html…") is parsed as the final
            // answer, so the tool never runs and the model only appears to act.
            Ok(ChatStreamItem::Reasoning(thinking)) => {
                self.agent_reasoning_buffer.push_str(&thinking);
                if let Some(last) = self.messages.last_mut() {
                    if last.role == ChatRole::Thought {
                        last.text.push_str(&thinking);
                    }
                }
                self.scroll_to_bottom();
                ctx.notify();
            }
            // A native tool call, streamed in fragments. Accumulated silently —
            // it becomes a proper tool step once the arguments are complete.
            Ok(ChatStreamItem::ToolCall { name, arguments }) => {
                let (call_name, call_args) = self
                    .agent_native_call
                    .get_or_insert_with(|| (None, String::new()));
                if name.is_some() {
                    *call_name = name;
                }
                call_args.push_str(&arguments);
            }
            // The step is settled in `on_agent_step_end`.
            Ok(ChatStreamItem::Done) => {}
            Err(e) => {
                let message = e.to_string();
                // The server doesn't take the `tools` field (llama-server without
                // `--jinja`). That's a capability mismatch, not a failed turn:
                // remember it, drop the field, and run the step again on the text
                // protocol — which is what every model had before.
                if self.native_tools != NativeToolSupport::Unsupported
                    && is_tools_unsupported_error(&message)
                {
                    log::info!("local chat: server rejected `tools`, falling back to the text protocol ({message})");
                    self.native_tools = NativeToolSupport::Unsupported;
                    // Drop this step's empty thought bubble; run_agent_step opens
                    // a fresh one, and two would stack up.
                    if let Some(last) = self.messages.last() {
                        if last.role == ChatRole::Thought && last.text.trim().is_empty() {
                            self.messages.pop();
                        }
                    }
                    self.run_agent_step(ctx);
                    ctx.notify();
                    return;
                }
                self.finish_agent_with_error(
                    turn,
                    format!("{}: {message}", self.active_backend_error_label()),
                    ctx,
                )
            }
        }
    }

    fn on_agent_step_end(&mut self, turn: u64, ctx: &mut ViewContext<Self>) {
        if turn != self.current_turn || !self.in_flight {
            return; // already errored out, stopped, or superseded
        }
        let mut reply = self.agent_step_buffer.trim().to_string();
        let budget_left = self.agent_step < MAX_AGENT_STEPS;

        // 1. A NATIVE tool call, on the model's own tool channel. Checked before
        //    anything else because a tool-native model often narrates in `content`
        //    AND calls in the same step — the call is the part that matters.
        if let Some(tool) = self.take_native_tool_call().filter(|_| budget_left) {
            self.native_tools = NativeToolSupport::Supported;
            // Record it in the TEXT protocol so the history stays in the one shape
            // the next step is asked to produce.
            self.agent_messages
                .push(ChatMessage::assistant(tool.to_tag()));
            let said = if reply.is_empty() {
                self.agent_reasoning_buffer.trim().to_string()
            } else {
                reply.clone()
            };
            self.finalize_thought(&said, true);
            if tool.requires_approval() && !self.auto_approve {
                self.pending_tool = Some(tool);
                self.scroll_to_bottom();
                ctx.notify();
            } else {
                self.start_tool(tool, ctx);
            }
            return;
        }

        // A harmony model (gpt-oss and friends) writes its analysis to one channel
        // and its answer to another; when it decides to ACT, everything it emits
        // can land outside `content`. So an empty answer here does not mean the
        // model said nothing — check the other channel before giving up.
        if reply.is_empty() {
            let thinking = self.agent_reasoning_buffer.trim().to_string();
            if !thinking.is_empty() {
                if local_agent::parse_tool_call(&thinking).is_some() {
                    // 2. It wrote the tool tag, just inside its reasoning. Take it
                    //    — the user asked for the work, not for the channel it was
                    //    announced on.
                    reply = thinking;
                } else if self.agent_nudges < MAX_AGENT_NUDGES && budget_left {
                    // 3. It thought out loud and stopped without ever opening its
                    //    answer channel. Feed the thinking back and ask for the
                    //    tool tag or the answer, rather than failing the turn.
                    self.agent_nudges += 1;
                    self.agent_messages
                        .push(ChatMessage::assistant(thinking.clone()));
                    self.agent_messages.push(ChatMessage::user(
                        "Continue. Reply with ONLY the tool tag for your next step, \
                         or your final answer as plain text. Do not stop at planning."
                            .to_string(),
                    ));
                    self.finalize_thought(&thinking, true);
                    self.run_agent_step(ctx);
                    ctx.notify();
                    return;
                } else {
                    // 4. Out of nudges. Its reasoning is the only thing it ever
                    //    produced, so show that instead of an error banner.
                    reply = thinking;
                }
            }
        }

        // Nothing on any channel: the endpoint accepted the request and closed
        // without generating. Fail with the reason (the plain-chat path does the
        // same) rather than looping on a model that told us nothing.
        if reply.is_empty() {
            let reason = self.empty_reply_error();
            self.finish_agent_with_error(turn, reason, ctx);
            return;
        }
        self.agent_messages
            .push(ChatMessage::assistant(reply.clone()));

        match local_agent::parse_tool_call(&reply) {
            Some(tool) if self.agent_step < MAX_AGENT_STEPS => {
                // Keep any reasoning the model wrote before the tag as a collapsed
                // thought; drop the thought entirely if it was only the tag.
                self.finalize_thought(&reply, true);
                // Side-effecting tools pause for the user unless AUTO is on. The
                // Allow/Deny prompt renders above the input (see `render`), so we
                // don't add a transcript entry yet — just remember the tool.
                if tool.requires_approval() && !self.auto_approve {
                    self.pending_tool = Some(tool);
                    self.scroll_to_bottom();
                    ctx.notify();
                    return;
                }
                self.start_tool(tool, ctx);
            }
            // The model SAID it would use a tool and then didn't — the exact
            // gpt-oss failure where "I'll use list_files" ends the turn with
            // nothing run. Narrating an intent is not a call, so push back
            // instead of accepting it as the answer.
            None if budget_left
                && self.agent_nudges < MAX_AGENT_NUDGES
                && local_agent::announced_tool_without_calling(&reply) =>
            {
                self.agent_nudges += 1;
                self.agent_messages.push(ChatMessage::user(
                    "You described a tool instead of calling one, so nothing ran. \
                     Call it now: reply with ONLY the tool tag and no other text, \
                     for example <tool:list_files path=\".\"/>."
                        .to_string(),
                ));
                self.finalize_thought(&reply, true);
                self.run_agent_step(ctx);
                ctx.notify();
            }
            _ => {
                // No tool call (or the step budget is spent): the reply is the
                // final answer. Keep the model's reasoning as its own collapsed
                // step and give the answer a bubble of its own — folding both
                // into one entry is why every reply read as a "Thought".
                let reasoning = self.agent_reasoning_buffer.trim().to_string();
                self.finalize_answer(&reasoning, &reply);
                if self.agent_step >= MAX_AGENT_STEPS {
                    if let Some(last) = self.messages.last_mut() {
                        if last.role == ChatRole::Assistant && last.text.trim().is_empty() {
                            last.text = "(reached the tool-step limit)".to_string();
                        }
                    }
                }
                self.in_flight = false;
                self.refresh_ai_mode();
                self.scroll_to_bottom();
                self.persist_mempalace(ctx, false);
                self.flush_pending_steering(ctx);
                ctx.notify();
            }
        }
    }

    /// Settle the streaming thought bubble for a finished step. When the step
    /// ended in a tool call, keep the pre-tool reasoning as a collapsed thought
    /// (or drop it if empty). Otherwise the reply *is* the final answer, so the
    /// bubble becomes a normal (markdown) assistant message.
    fn finalize_thought(&mut self, reply: &str, had_tool: bool) {
        let Some(idx) = self
            .messages
            .iter()
            .rposition(|m| m.role == ChatRole::Thought)
        else {
            return;
        };
        if had_tool {
            let visible = local_agent::strip_tool_calls(reply);
            if visible.is_empty() {
                self.messages.remove(idx);
            } else {
                let entry = &mut self.messages[idx];
                entry.text = visible;
                entry.collapsed = true;
                entry.status = StepStatus::Ok;
            }
        } else {
            let entry = &mut self.messages[idx];
            entry.role = ChatRole::Assistant;
            entry.text = reply.to_string();
            entry.collapsed = false;
            entry.status = StepStatus::Ok;
        }
    }

    /// Settle the streaming bubble for a step that ended in the FINAL answer.
    ///
    /// A thinking model produces two different things — the private analysis and
    /// the answer — and they were being written into the same entry, so the
    /// transcript showed the model's musings under "Genesi AI" as if that were
    /// the reply. They now become two entries: a collapsed thought, then the
    /// answer. When there is no reasoning (or it IS the answer, which happens
    /// when the loop falls back to it) the single bubble is kept.
    fn finalize_answer(&mut self, reasoning: &str, answer: &str) {
        let Some(idx) = self
            .messages
            .iter()
            .rposition(|m| m.role == ChatRole::Thought)
        else {
            return;
        };
        let split = !reasoning.is_empty() && reasoning != answer;
        if !split {
            let entry = &mut self.messages[idx];
            entry.role = ChatRole::Assistant;
            entry.text = answer.to_string();
            entry.collapsed = false;
            entry.status = StepStatus::Ok;
            return;
        }

        let thought = &mut self.messages[idx];
        thought.text = reasoning.to_string();
        thought.collapsed = true;
        thought.status = StepStatus::Ok;
        self.messages.push(ChatEntry::prose(
            ChatRole::Assistant,
            answer.to_string(),
            None,
        ));
    }

    fn finish_agent_with_error(&mut self, turn: u64, message: String, ctx: &mut ViewContext<Self>) {
        if turn != self.current_turn {
            return;
        }
        self.in_flight = false;
        // Drop a trailing empty thought/assistant placeholder if nothing useful
        // streamed into it.
        if let Some(last) = self.messages.last() {
            if matches!(last.role, ChatRole::Assistant | ChatRole::Thought)
                && last.text.trim().is_empty()
            {
                self.messages.pop();
            }
        }
        self.error = Some(message);
        self.persist_mempalace(ctx, false);
        ctx.notify();
    }

    /// Snapshot a write/edit *before* it runs: `(path, added, removed,
    /// original_content)`. Returns `None` for read tools. Best-effort — the line
    /// stats drive the `+N −N` diff card, the original drives Undo.
    fn capture_edit(&self, tool: &AgentTool) -> Option<(String, u32, u32, Option<String>)> {
        let root = self.agent_root.as_ref()?;
        let read_original = |path: &str| {
            local_agent::resolve_in_project(root, path)
                .and_then(|p| std::fs::read_to_string(p).ok())
        };
        match tool {
            AgentTool::WriteFile { path, content } => {
                let original = read_original(path);
                let removed = original.as_deref().map(count_lines).unwrap_or(0);
                Some((path.clone(), count_lines(content), removed, original))
            }
            AgentTool::EditFile {
                path,
                search,
                replace,
            } => {
                let original = read_original(path);
                Some((
                    path.clone(),
                    count_lines(replace),
                    count_lines(search),
                    original,
                ))
            }
            _ => None,
        }
    }

    fn read_project_file_after_edit(&self, path: &str) -> Option<String> {
        let root = self.agent_root.as_ref()?;
        local_agent::resolve_in_project(root, path).and_then(|p| std::fs::read_to_string(p).ok())
    }

    /// Record a successful edit for the review bar. Re-editing the same file keeps
    /// the EARLIEST original (so Undo restores the pre-turn state) and refreshes
    /// the displayed stats.
    fn record_edit(
        &mut self,
        path: String,
        added: u32,
        removed: u32,
        original: Option<String>,
        diff_preview: Vec<DiffPreviewLine>,
        ctx: &mut ViewContext<Self>,
    ) {
        self.update_open_editor_diff(&path, Some(original.as_deref().unwrap_or("")), ctx);

        if let Some(existing) = self.pending_edits.iter_mut().find(|e| e.path == path) {
            existing.added = added;
            existing.removed = removed;
            existing.diff_preview = diff_preview;
        } else {
            self.pending_edits.push(PendingEdit {
                path,
                added,
                removed,
                original,
                diff_preview,
            });
        }

        if self.selected_review_path.is_none() {
            self.selected_review_path = self.pending_edits.first().map(|edit| edit.path.clone());
        }
    }

    fn update_open_editor_diff(
        &self,
        relative_path: &str,
        original: Option<&str>,
        ctx: &mut ViewContext<Self>,
    ) {
        let Some(root) = self.agent_root.as_ref() else {
            return;
        };
        let Some(target) = local_agent::resolve_in_project(root, relative_path) else {
            return;
        };
        let Some(window_id) = ctx.windows().active_window() else {
            return;
        };
        let Some(editors) = ctx.views_of_type::<LocalCodeEditorView>(window_id) else {
            return;
        };

        for editor in editors {
            let matches = editor.as_ref(ctx).file_path().is_some_and(|path| {
                path == target
                    || path
                        .canonicalize()
                        .ok()
                        .zip(target.canonicalize().ok())
                        .is_some_and(|(path, target)| path == target)
            });
            if !matches {
                continue;
            }
            editor.update(ctx, |editor, ctx| match original {
                Some(original) => editor.show_pending_agent_diff(original, ctx),
                None => editor.keep_pending_agent_diff(ctx),
            });
        }
    }

    /// Tell the file tree to rescan after the agent touched the project.
    ///
    /// The tree only catches up on outside changes when it becomes active, so a
    /// file the agent created stayed invisible until the user switched to search
    /// and back. The agent writes straight to disk, which is exactly "behind the
    /// tree's back".
    fn refresh_file_tree(&self, ctx: &mut ViewContext<Self>) {
        let Some(window_id) = ctx.windows().active_window() else {
            return;
        };
        let Some(trees) = ctx.views_of_type::<FileTreeView>(window_id) else {
            return;
        };
        for tree in trees {
            tree.update(ctx, |tree, ctx| tree.refresh_local_directories(ctx));
        }
    }

    /// Re-apply every unreviewed edit's baseline to the editors that are open.
    ///
    /// The baseline is state of the editor VIEW, not of the file, so a file that
    /// gets closed and reopened comes back with no diff even though the change
    /// has not been reviewed yet. The panel is the one that still knows the
    /// pre-edit content, so it puts it back whenever a file finishes loading.
    pub fn restore_pending_diffs(&self, ctx: &mut ViewContext<Self>) {
        if self.pending_edits.is_empty() {
            return;
        }
        for edit in &self.pending_edits {
            let original = edit.original.clone().unwrap_or_default();
            self.update_open_editor_diff(&edit.path, Some(&original), ctx);
        }
    }

    /// Begin running an (already-approved) tool: reads run inline; run_command
    /// spawns. The loop continues in [`Self::finish_tool`].
    fn start_tool(&mut self, tool: AgentTool, ctx: &mut ViewContext<Self>) {
        self.agent_tool_summary = tool.summary();
        self.agent_step += 1;
        let turn = self.current_turn;

        match &tool {
            AgentTool::RunCommand { command } => {
                // A terminal block: the command runs in the project root and its
                // output is shown like an integrated terminal (`$ cmd` + output).
                self.messages.push(ChatEntry {
                    role: ChatRole::Command,
                    text: String::new(),
                    context_label: None,
                    tool_title: None,
                    command: Some(command.clone()),
                    collapsed: false,
                    status: StepStatus::Running,
                    diff_stat: None,
                    diff_preview: None,
                });
                self.scroll_to_bottom();
                ctx.notify();

                let command = command.clone();
                let root = self.agent_root.clone();
                ctx.spawn(
                    async move {
                        match root {
                            Some(root) => local_agent::run_command(&root, &command).await,
                            None => "error: no project is open.".to_string(),
                        }
                    },
                    move |me, result, ctx| me.finish_tool(turn, "run_command", result, ctx),
                );
            }
            _ => {
                // For a write/edit, snapshot the file first so we can show a
                // `+N −N` diff card and later undo it. Reads return None here.
                let edit_meta = self.capture_edit(&tool);
                let (title, diff_stat) = match &edit_meta {
                    Some((path, added, removed, _)) => (path.clone(), Some((*added, *removed))),
                    None => (self.agent_tool_summary.clone(), None),
                };
                // A read tool: collapsed by default, showing just its one-line
                // summary; an edit shows the file card. The detail is one click away.
                self.messages.push(ChatEntry {
                    role: ChatRole::Tool,
                    text: String::new(),
                    context_label: None,
                    tool_title: Some(title),
                    command: None,
                    collapsed: true,
                    status: StepStatus::Running,
                    diff_stat,
                    diff_preview: None,
                });
                let result = match self.agent_root.clone() {
                    Some(root) => local_agent::run_local_tool(&root, &tool),
                    None => "error: no project is open, so I can't read files.".to_string(),
                };
                // Record a successful edit for the review bar (Undo all / Keep).
                if let Some((path, added, removed, original)) = edit_meta {
                    if !result.trim_start().starts_with("error") {
                        let current = self.read_project_file_after_edit(&path).unwrap_or_default();
                        let diff_preview =
                            build_diff_preview(original.as_deref().unwrap_or(""), &current);
                        self.record_edit(
                            path.clone(),
                            added,
                            removed,
                            original,
                            diff_preview.clone(),
                            ctx,
                        );
                        if let Some(last) = self.messages.last_mut() {
                            last.diff_preview = Some(diff_preview);
                        }
                    }
                }
                self.finish_tool(turn, tool.name(), result, ctx);
            }
        }
    }

    /// Settle a finished tool: show its result, feed the full result back to the
    /// model, and take the next agent step.
    fn finish_tool(
        &mut self,
        turn: u64,
        tool_name: &str,
        result: String,
        ctx: &mut ViewContext<Self>,
    ) {
        if turn != self.current_turn {
            return; // the user stopped or started a new turn while this ran
        }
        let is_error = result.trim_start().starts_with("error");
        if let Some(last) = self.messages.last_mut() {
            match last.role {
                // The terminal block shows the full (already-capped) output.
                ChatRole::Command => last.text = result.clone(),
                ChatRole::Tool => last.text = Self::tool_preview(&result),
                _ => {}
            }
            last.status = if is_error {
                StepStatus::Error
            } else {
                StepStatus::Ok
            };
            // The line counts were measured BEFORE the tool ran, from what the
            // edit was going to do. When it fails — an edit_file whose SEARCH
            // block didn't match is the common one — leaving them on screen
            // draws a file card claiming "+53 −122" for a file that was never
            // touched. That is the "it says it edited but nothing changed"
            // report: the write really didn't happen, the UI just said it did.
            if is_error {
                last.diff_stat = None;
                last.diff_preview = None;
            }
        }
        // A tool that touched the project needs the tree to notice. run_command
        // counts too: `npm create vite@latest` produces a whole directory the
        // tree would otherwise never show.
        if !is_error && matches!(tool_name, "write_file" | "edit_file" | "run_command") {
            self.refresh_file_tree(ctx);
        }
        self.agent_messages.push(ChatMessage::user(format!(
            "TOOL RESULT ({tool_name}):\n{result}"
        )));
        self.scroll_to_bottom();
        self.persist_mempalace(ctx, false);
        ctx.notify();
        self.run_agent_step(ctx);
    }

    /// Interrupt the in-flight generation / agent loop. The detached streams
    /// can't be cancelled directly, so bump the turn id (their callbacks then
    /// no-op) and settle whatever was on screen.
    fn stop_turn(&mut self, ctx: &mut ViewContext<Self>) {
        if !self.in_flight {
            return;
        }
        self.current_turn += 1;
        self.in_flight = false;
        self.pending_tool = None;
        // Stop means stop: a correction queued for a turn the user just
        // abandoned must not resurface on the next one.
        self.steering.clear();

        let trailing = self
            .messages
            .last()
            .map(|m| (m.role, m.text.trim().is_empty()));
        match trailing {
            Some((ChatRole::Thought, true)) | Some((ChatRole::Assistant, true)) => {
                self.messages.pop();
            }
            Some((ChatRole::Thought, false)) => {
                if let Some(last) = self.messages.last_mut() {
                    last.role = ChatRole::Assistant;
                    last.collapsed = false;
                    last.status = StepStatus::Ok;
                }
            }
            Some((ChatRole::Command, _)) | Some((ChatRole::Tool, _)) => {
                if let Some(last) = self.messages.last_mut() {
                    if last.status == StepStatus::Running {
                        last.status = StepStatus::Denied;
                    }
                }
            }
            _ => {}
        }
        self.error = None;
        self.refresh_ai_mode();
        self.persist_mempalace(ctx, false);
        ctx.notify();
    }

    /// A short preview of a read-tool result for the collapsed step; the full
    /// result is what goes back to the model.
    fn tool_preview(result: &str) -> String {
        const MAX_LINES: usize = 8;
        const MAX_CHARS: usize = 600;
        let mut preview: String = result
            .lines()
            .take(MAX_LINES)
            .collect::<Vec<_>>()
            .join("\n");
        if preview.chars().count() > MAX_CHARS {
            preview = preview.chars().take(MAX_CHARS).collect();
        }
        if result.lines().count() > MAX_LINES || result.chars().count() > MAX_CHARS {
            preview.push_str("\n…");
        }
        preview
    }

    fn on_stream_item(
        &mut self,
        turn: u64,
        item: Result<ChatStreamItem>,
        ctx: &mut ViewContext<Self>,
    ) {
        if turn != self.current_turn {
            return; // a stopped / superseded turn
        }
        match item {
            // Plain chat has no tool parsing, so a thinking model's reasoning is
            // simply shown — seeing it beats an empty bubble while it works.
            Ok(ChatStreamItem::Token(token)) | Ok(ChatStreamItem::Reasoning(token)) => {
                if let Some(last) = self.messages.last_mut() {
                    if last.role == ChatRole::Assistant {
                        last.text.push_str(&token);
                    }
                }
                self.scroll_to_bottom();
                ctx.notify();
            }
            // Chat mode runs no tools, so a model that reaches for one has only
            // its own reasoning to show — nothing to render here.
            Ok(ChatStreamItem::ToolCall { .. }) => {}
            Ok(ChatStreamItem::Done) => {
                self.in_flight = false;
                // tokens/s and activity likely changed — refresh the badge.
                self.refresh_ai_mode();
                self.scroll_to_bottom();
                self.persist_mempalace(ctx, false);
                self.flush_pending_steering(ctx);
                ctx.notify();
            }
            Err(e) => {
                self.in_flight = false;
                // Drop the empty assistant placeholder if nothing streamed.
                if let Some(last) = self.messages.last() {
                    if last.role == ChatRole::Assistant && last.text.is_empty() {
                        self.messages.pop();
                    }
                }
                self.error = Some(format!("{}: {e}", self.active_backend_error_label()));
                self.persist_mempalace(ctx, false);
                ctx.notify();
            }
        }
    }

    // ── rendering ────────────────────────────────────────────────────────────

    fn label_text(
        &self,
        appearance: &Appearance,
        text: impl Into<String>,
        size: f32,
        color: ColorU,
        soft_wrap: bool,
    ) -> Box<dyn Element> {
        self.styled_text(appearance, text, size, color, soft_wrap, false)
    }

    /// Like [`Self::label_text`] but in the monospace family — for tool output
    /// and terminal blocks, where columns and code need to line up.
    fn mono_text(
        &self,
        appearance: &Appearance,
        text: impl Into<String>,
        size: f32,
        color: ColorU,
        soft_wrap: bool,
    ) -> Box<dyn Element> {
        self.styled_text(appearance, text, size, color, soft_wrap, true)
    }

    fn styled_text(
        &self,
        appearance: &Appearance,
        text: impl Into<String>,
        size: f32,
        color: ColorU,
        soft_wrap: bool,
        monospace: bool,
    ) -> Box<dyn Element> {
        let family = if monospace {
            appearance.monospace_font_family()
        } else {
            appearance.ui_font_family()
        };
        appearance
            .ui_builder()
            .wrappable_text(text.into(), soft_wrap)
            .with_style(UiComponentStyles {
                font_family_id: Some(family),
                font_size: Some(size),
                font_color: Some(color),
                ..Default::default()
            })
            .build()
            .finish()
    }

    /// Render markdown (headings, lists, **bold**, `code`, fenced blocks) into a
    /// laid-out element so the assistant's replies read like a real AI IDE
    /// instead of showing raw `#`/`*`/backtick characters.
    fn markdown_text(
        &self,
        appearance: &Appearance,
        text: &str,
        color: ColorU,
    ) -> Box<dyn Element> {
        let formatted = parse_markdown(text).unwrap_or_else(|_| {
            FormattedText::new([FormattedTextLine::Line(vec![
                FormattedTextFragment::plain_text(text.to_string()),
            ])])
        });
        Box::new(
            FormattedTextElement::new(
                formatted,
                BODY_FONT_SIZE,
                appearance.ui_font_family(),
                appearance.monospace_font_family(),
                color,
                Default::default(),
            )
            .set_selectable(true),
        )
    }

    fn chip_content(
        &self,
        appearance: &Appearance,
        label: String,
        icon_path: Option<&'static str>,
        selected: bool,
        enabled: bool,
    ) -> Box<dyn Element> {
        let text_color: ColorU = if enabled {
            if selected {
                ColorU::new(215, 248, 234, 255)
            } else {
                ColorU::new(228, 231, 236, 255)
            }
        } else {
            ColorU::new(132, 137, 145, 255)
        };
        let background = if selected {
            ColorU::new(15, 143, 106, 44)
        } else if enabled {
            ColorU::new(255, 255, 255, 14)
        } else {
            ColorU::new(255, 255, 255, 8)
        };
        let border = if selected {
            green_soft()
        } else {
            ColorU::new(255, 255, 255, 32)
        };
        let icon_color = if selected { genesi_green() } else { text_color };

        let mut row = Flex::row().with_cross_axis_alignment(CrossAxisAlignment::Center);
        if let Some(path) = icon_path {
            row.add_child(
                Container::new(
                    ConstrainedBox::new(Icon::new(path, icon_color).finish())
                        .with_width(12.)
                        .with_height(12.)
                        .finish(),
                )
                .with_margin_right(5.)
                .finish(),
            );
        }
        row.add_child(self.label_text(appearance, label, CHIP_FONT_SIZE, text_color, false));

        Container::new(row.finish())
            .with_horizontal_padding(8.)
            .with_vertical_padding(5.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(7.)))
            .with_border(Border::all(1.).with_border_color(border))
            .with_background_color(background)
            .with_margin_right(6.)
            .with_margin_top(2.)
            .finish()
    }

    fn chip_with_icon(
        &self,
        appearance: &Appearance,
        label: String,
        icon_path: Option<&'static str>,
        action: LocalAiChatAction,
        selected: bool,
        enabled: bool,
    ) -> Box<dyn Element> {
        let content = self.chip_content(appearance, label, icon_path, selected, enabled);
        if !enabled {
            return content;
        }

        EventHandler::new(content)
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    fn chip(
        &self,
        appearance: &Appearance,
        label: String,
        action: LocalAiChatAction,
        selected: bool,
    ) -> Box<dyn Element> {
        self.chip_with_icon(appearance, label, None, action, selected, true)
    }

    fn workspace_chip(
        &self,
        appearance: &Appearance,
        label: String,
        icon_path: Option<&'static str>,
        action: WorkspaceAction,
        selected: bool,
    ) -> Box<dyn Element> {
        EventHandler::new(self.chip_content(appearance, label, icon_path, selected, true))
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    fn project_graph_colors(&self, appearance: &Appearance) -> ProjectGraphColors {
        let theme = appearance.theme();
        ProjectGraphColors {
            background: theme.background().into(),
            grid: theme.outline().with_opacity(72).into(),
            page: theme.terminal_colors().normal.blue.into(),
            router: theme.terminal_colors().normal.magenta.into(),
            endpoint: theme.terminal_colors().normal.green.into(),
        }
    }

    fn canvas_node_icon(kind: CanvasNodeKind) -> CoreIcon {
        match kind {
            CanvasNodeKind::Page => CoreIcon::LayoutAlt01,
            CanvasNodeKind::Router => CoreIcon::Dataflow04,
            CanvasNodeKind::Endpoint => CoreIcon::Globe,
        }
    }

    fn render_forge_canvas_node(
        &self,
        appearance: &Appearance,
        node: &CanvasNode,
        selected: bool,
        zoom: f32,
        // Edge degree per node id, computed once per render. Deriving it here
        // meant rescanning every edge for every node drawn.
        degrees: &HashMap<&str, usize>,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let accent = self.canvas_node_accent(appearance, node.kind);
        let background = theme
            .surface_1()
            .blend(&ThemeFill::Solid(accent).with_opacity(if selected { 34 } else { 14 }));
        let icon_background = theme
            .surface_1()
            .blend(&ThemeFill::Solid(accent).with_opacity(44));
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let title = self.label_text(
            appearance,
            node.title.clone(),
            (14. * zoom).clamp(10., 17.),
            theme.main_text_color(theme.background()).into(),
            false,
        );
        let icon = ConstrainedBox::new(
            Container::new(
                Self::canvas_node_icon(node.kind)
                    .to_warpui_icon(ThemeFill::Solid(accent))
                    .finish(),
            )
            .with_uniform_padding((7. * zoom).clamp(5., 9.))
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9. * zoom)))
            .with_background(icon_background)
            .with_border(
                Border::all(1.).with_border_fill(ThemeFill::Solid(accent).with_opacity(36)),
            )
            .finish(),
        )
        .with_width(32. * zoom)
        .with_height(32. * zoom)
        .finish();
        let detail = match node.kind {
            CanvasNodeKind::Page => node
                .estimated_load_ms
                .map(|load| format!("~{load} ms estimated load"))
                .unwrap_or_else(|| "Frontend page".to_string()),
            CanvasNodeKind::Router => {
                let count = degrees.get(node.id.as_str()).copied().unwrap_or(0);
                format!("{count} route handlers")
            }
            CanvasNodeKind::Endpoint => format!(
                "{} request · {}",
                node.method.as_deref().unwrap_or("ANY"),
                node.source.display()
            ),
        };
        let status = match node.kind {
            CanvasNodeKind::Page => "Page",
            CanvasNodeKind::Router => "Router",
            CanvasNodeKind::Endpoint => node.method.as_deref().unwrap_or("Endpoint"),
        };
        let content = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(
                Flex::row()
                    .with_cross_axis_alignment(CrossAxisAlignment::Center)
                    .with_child(icon)
                    .with_child(
                        Container::new(Expanded::new(1., title).finish())
                            .with_margin_left(10. * zoom)
                            .finish(),
                    )
                    .finish(),
            )
            .with_child(
                Container::new(
                    Flex::row()
                        .with_cross_axis_alignment(CrossAxisAlignment::Center)
                        .with_child(
                            ConstrainedBox::new(
                                Container::new(Empty::new().finish())
                                    .with_background_color(accent)
                                    .with_corner_radius(CornerRadius::with_all(Radius::Percentage(
                                        50.,
                                    )))
                                    .finish(),
                            )
                            .with_width(5. * zoom)
                            .with_height(5. * zoom)
                            .finish(),
                        )
                        .with_child(
                            Container::new(self.label_text(
                                appearance,
                                detail,
                                (11. * zoom).clamp(8., 13.),
                                muted,
                                false,
                            ))
                            .with_margin_left(7. * zoom)
                            .finish(),
                        )
                        .finish(),
                )
                .with_margin_top(9. * zoom)
                .finish(),
            )
            .with_child(
                Container::new(self.label_text(
                    appearance,
                    status.to_string(),
                    (10. * zoom).clamp(8., 12.),
                    accent,
                    false,
                ))
                .with_margin_top(8. * zoom)
                .with_uniform_padding(4. * zoom)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(7. * zoom)))
                .with_background(
                    theme
                        .surface_1()
                        .blend(&ThemeFill::Solid(accent).with_opacity(34)),
                )
                .finish(),
            )
            .finish();

        ConstrainedBox::new(
            Container::new(content)
                .with_uniform_padding(14. * zoom)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(14. * zoom)))
                .with_background(background)
                .with_border(
                    Border::all(if selected { 2. } else { 1.5 }).with_border_fill(
                        ThemeFill::Solid(accent).with_opacity(if selected { 90 } else { 50 }),
                    ),
                )
                .finish(),
        )
        .with_width(FORGE_NODE_WIDTH * zoom)
        .with_height(FORGE_NODE_HEIGHT * zoom)
        .finish()
    }

    fn render_canvas_palette_item(
        &self,
        appearance: &Appearance,
        node: &CanvasNode,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let accent = self.canvas_node_accent(appearance, node.kind);
        let selected = self.selected_canvas_node.as_deref() == Some(node.id.as_str());
        let subtitle = match node.kind {
            CanvasNodeKind::Endpoint => node.method.as_deref().unwrap_or("ANY").to_string(),
            _ => node.kind.label().to_string(),
        };
        let action = WorkspaceAction::SelectGenesiCanvasNode(node.id.clone());
        EventHandler::new(
            Container::new(
                Flex::row()
                    .with_cross_axis_alignment(CrossAxisAlignment::Center)
                    .with_child(
                        ConstrainedBox::new(
                            Container::new(
                                Self::canvas_node_icon(node.kind)
                                    .to_warpui_icon(ThemeFill::Solid(accent))
                                    .finish(),
                            )
                            .with_uniform_padding(6.)
                            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(7.)))
                            .with_background(
                                theme
                                    .surface_1()
                                    .blend(&ThemeFill::Solid(accent).with_opacity(42)),
                            )
                            .finish(),
                        )
                        .with_width(28.)
                        .with_height(28.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(
                            Flex::column()
                                .with_child(self.label_text(
                                    appearance,
                                    node.title.clone(),
                                    12.,
                                    theme.main_text_color(theme.background()).into(),
                                    false,
                                ))
                                .with_child(self.label_text(
                                    appearance,
                                    subtitle,
                                    9.,
                                    theme.disabled_text_color(theme.background()).into(),
                                    false,
                                ))
                                .finish(),
                        )
                        .with_margin_left(9.)
                        .finish(),
                    )
                    .finish(),
            )
            .with_uniform_padding(8.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9.)))
            .with_background(if selected {
                theme
                    .surface_1()
                    .blend(&ThemeFill::Solid(accent).with_opacity(30))
            } else {
                theme.surface_1()
            })
            .with_border(Border::all(1.).with_border_fill(if selected {
                ThemeFill::Solid(accent).with_opacity(63)
            } else {
                theme.outline()
            }))
            .finish(),
        )
        .on_left_mouse_down(move |ctx, _, _| {
            ctx.dispatch_typed_action(action.clone());
            DispatchEventResult::StopPropagation
        })
        .finish()
    }

    fn render_project_canvas_palette(
        &self,
        appearance: &Appearance,
        graph: &ProjectCanvasGraph,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let mut nodes = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);
        for (kind, section) in [
            (CanvasNodeKind::Page, "FRONTEND PAGES"),
            (CanvasNodeKind::Router, "BACKEND ROUTERS"),
            (CanvasNodeKind::Endpoint, "API ENDPOINTS"),
        ] {
            let section_nodes = graph
                .nodes
                .iter()
                .filter(|node| node.kind == kind)
                .collect::<Vec<_>>();
            if section_nodes.is_empty() {
                continue;
            }
            nodes.add_child(
                Container::new(self.label_text(
                    appearance,
                    section.to_string(),
                    9.,
                    theme.disabled_text_color(theme.background()).into(),
                    false,
                ))
                .with_margin_top(12.)
                .with_margin_bottom(5.)
                .finish(),
            );
            for node in section_nodes {
                nodes.add_child(
                    Container::new(self.render_canvas_palette_item(appearance, node))
                        .with_margin_bottom(6.)
                        .finish(),
                );
            }
        }

        Container::new(
            Flex::column()
                .with_main_axis_size(MainAxisSize::Max)
                .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                .with_child(self.label_text(
                    appearance,
                    "Nodes".to_string(),
                    16.,
                    theme.main_text_color(theme.background()).into(),
                    false,
                ))
                .with_child(
                    Container::new(self.label_text(
                        appearance,
                        "Explore the architecture discovered from source".to_string(),
                        11.,
                        theme.disabled_text_color(theme.background()).into(),
                        true,
                    ))
                    .with_margin_top(4.)
                    .finish(),
                )
                .with_child(
                    Expanded::new(
                        1.,
                        ClippedScrollable::vertical(
                            self.project_canvas_palette_scroll.clone(),
                            nodes.finish(),
                            ScrollbarWidth::Auto,
                            theme.disabled_ui_text_color().into(),
                            theme.active_ui_text_color().into(),
                            Fill::None,
                        )
                        .finish(),
                    )
                    .finish(),
                )
                .with_child(
                    Container::new(self.workspace_chip(
                        appearance,
                        "Auto-arrange".to_string(),
                        None,
                        WorkspaceAction::AutoArrangeGenesiCanvas,
                        false,
                    ))
                    .with_margin_top(8.)
                    .finish(),
                )
                .finish(),
        )
        .with_uniform_padding(14.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(12.)))
        .with_background(theme.surface_1())
        .with_border(Border::all(1.).with_border_fill(theme.outline()))
        .finish()
    }

    fn render_canvas_inspector_field(
        &self,
        appearance: &Appearance,
        label: &str,
        value: String,
        monospace: bool,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let value = if monospace {
            self.mono_text(
                appearance,
                value,
                12.,
                theme.main_text_color(theme.background()).into(),
                true,
            )
        } else {
            self.label_text(
                appearance,
                value,
                12.,
                theme.main_text_color(theme.background()).into(),
                true,
            )
        };
        Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(self.label_text(
                appearance,
                label.to_string(),
                11.,
                theme.disabled_text_color(theme.background()).into(),
                false,
            ))
            .with_child(
                Container::new(value)
                    .with_uniform_padding(10.)
                    .with_margin_top(6.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                    .with_background(theme.background())
                    .with_border(Border::all(1.).with_border_fill(theme.outline()))
                    .finish(),
            )
            .finish()
    }

    fn render_project_canvas_inspector(
        &self,
        appearance: &Appearance,
        graph: &ProjectCanvasGraph,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let selected = self
            .selected_canvas_node
            .as_deref()
            .and_then(|id| graph.node(id));
        let mut content = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);
        if let Some(node) = selected {
            let accent = self.canvas_node_accent(appearance, node.kind);
            content.add_child(
                Flex::row()
                    .with_cross_axis_alignment(CrossAxisAlignment::Center)
                    .with_child(
                        ConstrainedBox::new(
                            Container::new(
                                Self::canvas_node_icon(node.kind)
                                    .to_warpui_icon(ThemeFill::Solid(accent))
                                    .finish(),
                            )
                            .with_uniform_padding(8.)
                            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9.)))
                            .with_background(
                                theme
                                    .surface_1()
                                    .blend(&ThemeFill::Solid(accent).with_opacity(42)),
                            )
                            .finish(),
                        )
                        .with_width(36.)
                        .with_height(36.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(
                            Flex::column()
                                .with_child(self.label_text(
                                    appearance,
                                    node.title.clone(),
                                    16.,
                                    theme.main_text_color(theme.background()).into(),
                                    false,
                                ))
                                .with_child(self.label_text(
                                    appearance,
                                    format!(
                                        "Configure this {} node",
                                        node.kind.label().to_lowercase()
                                    ),
                                    11.,
                                    theme.disabled_text_color(theme.background()).into(),
                                    false,
                                ))
                                .finish(),
                        )
                        .with_margin_left(10.)
                        .finish(),
                    )
                    .finish(),
            );
            content.add_child(
                Container::new(
                    Flex::row()
                        .with_main_axis_alignment(MainAxisAlignment::SpaceEvenly)
                        .with_child(self.label_text(
                            appearance,
                            "Config".to_string(),
                            12.,
                            accent,
                            false,
                        ))
                        .with_child(self.label_text(
                            appearance,
                            "Source".to_string(),
                            12.,
                            theme.disabled_text_color(theme.background()).into(),
                            false,
                        ))
                        .finish(),
                )
                .with_padding_top(14.)
                .with_padding_bottom(10.)
                .with_margin_top(8.)
                .with_border(Border::bottom(2.).with_border_color(accent))
                .finish(),
            );

            // An endpoint on the Canvas is a thing you can *call*, not just read
            // about. This hands it to Probe already filled in.
            if node.kind == CanvasNodeKind::Endpoint {
                content.add_child(
                    Container::new(self.workspace_chip(
                        appearance,
                        "Send request in Probe".to_string(),
                        None,
                        WorkspaceAction::ProbeGenesiCanvasNode(node.id.clone()),
                        false,
                    ))
                    .with_margin_top(14.)
                    .finish(),
                );
            }

            if let Some(route) = &node.route {
                content.add_child(
                    Container::new(self.render_canvas_inspector_field(
                        appearance,
                        "Route",
                        route.clone(),
                        true,
                    ))
                    .with_margin_top(14.)
                    .finish(),
                );
            }
            if let Some(method) = &node.method {
                content.add_child(
                    Container::new(self.render_canvas_inspector_field(
                        appearance,
                        "HTTP method",
                        method.clone(),
                        true,
                    ))
                    .with_margin_top(12.)
                    .finish(),
                );
            }
            content.add_child(
                Container::new(self.render_canvas_inspector_field(
                    appearance,
                    "Source location",
                    format!("{}:{}", node.source.display(), node.line),
                    true,
                ))
                .with_margin_top(12.)
                .finish(),
            );
            if let Some(load) = node.estimated_load_ms {
                content.add_child(
                    Container::new(self.render_canvas_inspector_field(
                        appearance,
                        "Estimated load",
                        format!("~{load} ms · static estimate"),
                        false,
                    ))
                    .with_margin_top(12.)
                    .finish(),
                );
            }
            if let Some(body) = &node.request_body {
                content.add_child(
                    Container::new(self.render_canvas_inspector_field(
                        appearance,
                        "Required request body",
                        body.clone(),
                        true,
                    ))
                    .with_margin_top(12.)
                    .finish(),
                );
            }
            let dependencies = if node.dependencies.is_empty() {
                "No direct imports detected".to_string()
            } else {
                node.dependencies.join("\n")
            };
            content.add_child(
                Container::new(self.render_canvas_inspector_field(
                    appearance,
                    "Dependencies",
                    dependencies,
                    true,
                ))
                .with_margin_top(12.)
                .finish(),
            );

            let connections = graph
                .edges
                .iter()
                .filter(|edge| edge.from == node.id || edge.to == node.id)
                .collect::<Vec<_>>();
            if !connections.is_empty() {
                content.add_child(
                    Container::new(self.label_text(
                        appearance,
                        "Connections".to_string(),
                        11.,
                        theme.disabled_text_color(theme.background()).into(),
                        false,
                    ))
                    .with_margin_top(14.)
                    .with_margin_bottom(6.)
                    .finish(),
                );
                for edge in connections {
                    let target_id = if edge.from == node.id {
                        edge.to.clone()
                    } else {
                        edge.from.clone()
                    };
                    let target_title = graph
                        .node(&target_id)
                        .map(|target| target.title.clone())
                        .unwrap_or(target_id.clone());
                    content.add_child(
                        Container::new(self.workspace_chip(
                            appearance,
                            format!("{}  ·  {target_title}", edge.kind.label()),
                            None,
                            WorkspaceAction::SelectGenesiCanvasNode(target_id),
                            false,
                        ))
                        .with_margin_bottom(6.)
                        .finish(),
                    );
                }
            }
        } else {
            content.add_child(self.render_canvas_message(
                appearance,
                "No node selected",
                "Select a page, router, or endpoint to inspect its configuration.",
            ));
        }

        Container::new(
            ClippedScrollable::vertical(
                self.project_canvas_inspector_scroll.clone(),
                Container::new(content.finish())
                    .with_uniform_padding(14.)
                    .finish(),
                ScrollbarWidth::Auto,
                theme.disabled_ui_text_color().into(),
                theme.active_ui_text_color().into(),
                Fill::None,
            )
            .finish(),
        )
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(12.)))
        .with_background(theme.surface_1())
        .with_border(Border::all(1.).with_border_fill(theme.outline()))
        .finish()
    }

    fn render_project_graph_surface(
        &self,
        appearance: &Appearance,
        graph: &Arc<ProjectCanvasGraph>,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let colors = self.project_graph_colors(appearance);
        // Build elements only for nodes that can land on screen. The canvas
        // element already skips laying out and painting off-screen nodes, but
        // constructing their text and icons was still costing a full pass per
        // frame — which is what made dragging a large graph crawl.
        let mut degrees: HashMap<&str, usize> = HashMap::new();
        for edge in &graph.edges {
            *degrees.entry(edge.from.as_str()).or_default() += 1;
            *degrees.entry(edge.to.as_str()).or_default() += 1;
        }

        let viewport = self.project_canvas_viewport.get();
        let node_size = vec2f(FORGE_NODE_WIDTH, FORGE_NODE_HEIGHT) * self.project_canvas_zoom;
        // One node of slack on every side, so a node that is only just outside
        // is still built and scrolls in without a pop.
        let cull_bounds = (viewport.x() > 0. && viewport.y() > 0.)
            .then(|| RectF::new(-node_size, viewport + node_size * 2.));
        let nodes = graph
            .nodes
            .iter()
            .filter_map(|node| {
                let position = self.project_canvas_positions.get(&node.id).copied()?;
                let screen_origin = self.project_canvas_pan + position * self.project_canvas_zoom;
                if let Some(bounds) = cull_bounds {
                    if bounds
                        .intersection(RectF::new(screen_origin, node_size))
                        .is_none()
                    {
                        return None;
                    }
                }
                Some(ProjectGraphNodeElement::new(
                    node.id.clone(),
                    position,
                    self.render_forge_canvas_node(
                        appearance,
                        node,
                        self.selected_canvas_node.as_deref() == Some(node.id.as_str()),
                        self.project_canvas_zoom,
                        &degrees,
                    ),
                ))
            })
            .collect::<Vec<_>>();
        let canvas = ProjectGraphCanvas::new(
            graph.clone(),
            nodes,
            self.project_canvas_pan,
            self.project_canvas_zoom,
            self.project_canvas_drag.is_some(),
            colors,
            self.project_canvas_viewport.clone(),
            self.project_canvas_positions.clone(),
        )
        .finish();
        let minimap = ConstrainedBox::new(
            Container::new(
                Flex::column()
                    .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                    .with_child(self.label_text(
                        appearance,
                        "Canvas Mini-map".to_string(),
                        10.,
                        theme.disabled_text_color(theme.background()).into(),
                        false,
                    ))
                    .with_child(
                        Container::new(
                            ConstrainedBox::new(
                                ProjectGraphMinimap::new(
                                    graph.clone(),
                                    self.project_canvas_positions.clone(),
                                    colors,
                                )
                                .finish(),
                            )
                            .with_width(168.)
                            .with_height(86.)
                            .finish(),
                        )
                        .with_margin_top(6.)
                        .finish(),
                    )
                    .finish(),
            )
            .with_uniform_padding(10.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(10.)))
            .with_background(theme.surface_1())
            .with_border(Border::all(1.).with_border_fill(theme.outline()))
            .finish(),
        )
        .with_width(190.)
        .with_height(128.)
        .finish();
        let zoom_controls = ConstrainedBox::new(
            Container::new(
                Flex::column()
                    .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                    .with_child(self.workspace_chip(
                        appearance,
                        "Fit".to_string(),
                        None,
                        WorkspaceAction::FitGenesiCanvas,
                        false,
                    ))
                    .with_child(
                        Container::new(self.workspace_chip(
                            appearance,
                            "+".to_string(),
                            None,
                            WorkspaceAction::ZoomGenesiCanvas(0.1),
                            false,
                        ))
                        .with_margin_top(6.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(self.workspace_chip(
                            appearance,
                            "−".to_string(),
                            None,
                            WorkspaceAction::ZoomGenesiCanvas(-0.1),
                            false,
                        ))
                        .with_margin_top(6.)
                        .finish(),
                    )
                    .finish(),
            )
            .finish(),
        )
        .with_width(56.)
        .with_height(112.)
        .finish();

        let mut stack = Stack::new();
        stack.add_child(canvas);
        stack.add_positioned_child(
            minimap,
            OffsetPositioning::offset_from_parent(
                vec2f(16., -16.),
                ParentOffsetBounds::WindowByPosition,
                ParentAnchor::BottomLeft,
                ChildAnchor::BottomLeft,
            ),
        );
        stack.add_positioned_child(
            zoom_controls,
            OffsetPositioning::offset_from_parent(
                vec2f(-16., -16.),
                ParentOffsetBounds::WindowByPosition,
                ParentAnchor::BottomRight,
                ChildAnchor::BottomRight,
            ),
        );

        Container::new(Clipped::new(stack.finish()).finish())
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(12.)))
            .with_background(theme.background())
            .with_border(Border::all(1.).with_border_fill(theme.outline()))
            .finish()
    }

    fn render_project_canvas_header(
        &self,
        appearance: &Appearance,
        graph: Option<&ProjectCanvasGraph>,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let project_name = graph
            .map(|graph| graph.project_name.clone())
            .unwrap_or_else(|| "Project".to_string());
        let subtitle = graph
            .map(|graph| {
                format!(
                    "Visual architecture of {project_name} · {}",
                    graph.kind.label()
                )
            })
            .unwrap_or_else(|| "Visual architecture from project source".to_string());
        let project_summary = graph
            .map(|graph| {
                let stacks = if graph.stacks.is_empty() {
                    "Source detected".to_string()
                } else {
                    graph.stacks.join(" · ")
                };
                format!("{project_name}  ·  {stacks}")
            })
            .unwrap_or(project_name);

        Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_child(
                ConstrainedBox::new(
                    Container::new(
                        CoreIcon::Lightning
                            .to_warpui_icon(ThemeFill::Solid(
                                theme.terminal_colors().normal.green.into(),
                            ))
                            .finish(),
                    )
                    .with_uniform_padding(8.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9.)))
                    .with_background(
                        theme.surface_1().blend(
                            &ThemeFill::Solid(theme.terminal_colors().normal.green.into())
                                .with_opacity(42),
                        ),
                    )
                    .finish(),
                )
                .with_width(34.)
                .with_height(34.)
                .finish(),
            )
            .with_child(
                Container::new(
                    Flex::column()
                        .with_child(self.label_text(
                            appearance,
                            "Project Canvas".to_string(),
                            18.,
                            theme.main_text_color(theme.background()).into(),
                            false,
                        ))
                        .with_child(self.label_text(
                            appearance,
                            subtitle,
                            11.,
                            theme.disabled_text_color(theme.background()).into(),
                            false,
                        ))
                        .finish(),
                )
                .with_margin_left(12.)
                .finish(),
            )
            .with_child(Expanded::new(1., Empty::new().finish()).finish())
            .with_child(
                Container::new(self.label_text(
                    appearance,
                    project_summary,
                    12.,
                    theme.main_text_color(theme.background()).into(),
                    false,
                ))
                .with_uniform_padding(10.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                .with_background(theme.surface_1())
                .with_border(Border::all(1.).with_border_fill(theme.outline()))
                .finish(),
            )
            .with_child(
                Container::new(self.workspace_chip(
                    appearance,
                    "Auto-arrange".to_string(),
                    None,
                    WorkspaceAction::AutoArrangeGenesiCanvas,
                    false,
                ))
                .with_margin_left(10.)
                .finish(),
            )
            .with_child(
                Container::new(self.workspace_chip(
                    appearance,
                    "Refresh".to_string(),
                    None,
                    WorkspaceAction::RefreshGenesiCanvas,
                    false,
                ))
                .with_margin_left(8.)
                .finish(),
            )
            .with_child(
                Container::new(self.workspace_chip(
                    appearance,
                    "Close".to_string(),
                    None,
                    WorkspaceAction::CloseGenesiCanvas,
                    false,
                ))
                .with_margin_left(8.)
                .finish(),
            )
            .finish()
    }

    pub fn render_project_canvas_workspace(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let graph = match &self.project_canvas_state {
            ProjectCanvasState::Ready(graph) => Some(graph.as_ref()),
            _ => None,
        };
        let mut root = Flex::column()
            .with_main_axis_size(MainAxisSize::Max)
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(
                Container::new(self.render_project_canvas_header(appearance, graph))
                    .with_padding_left(16.)
                    .with_padding_right(16.)
                    .with_padding_top(12.)
                    .with_padding_bottom(12.)
                    .finish(),
            );

        match &self.project_canvas_state {
            ProjectCanvasState::NoProject => root.add_child(
                Expanded::new(
                    1.,
                    Container::new(self.render_canvas_message(
                        appearance,
                        "No project detected",
                        "Open a project folder or focus a source file, then refresh Project Canvas.",
                    ))
                    .with_uniform_padding(120.)
                    .finish(),
                )
                .finish(),
            ),
            ProjectCanvasState::Loading(path) => root.add_child(
                Expanded::new(
                    1.,
                    Container::new(self.render_canvas_message(
                        appearance,
                        "Analyzing project…",
                        &format!(
                            "Scanning {} for pages, routers, endpoints, and source connections. Project code is never executed.",
                            path.display()
                        ),
                    ))
                    .with_uniform_padding(120.)
                    .finish(),
                )
                .finish(),
            ),
            ProjectCanvasState::Error(error) => root.add_child(
                Expanded::new(
                    1.,
                    Container::new(self.render_canvas_message(
                        appearance,
                        "Project analysis failed",
                        error,
                    ))
                    .with_uniform_padding(120.)
                    .finish(),
                )
                .finish(),
            ),
            ProjectCanvasState::Ready(graph) if graph.kind == ProjectKind::Unknown => {
                root.add_child(
                    Expanded::new(
                        1.,
                        Container::new(self.render_canvas_message(
                            appearance,
                            "No recognized project structure",
                            &format!(
                                "Scanned {} source files in {} but found no supported pages or endpoints.",
                                graph.files_scanned,
                                graph.root.display()
                            ),
                        ))
                        .with_uniform_padding(120.)
                        .finish(),
                    )
                    .finish(),
                );
            }
            ProjectCanvasState::Ready(graph) => {
                root.add_child(
                    Expanded::new(
                        1.,
                        Container::new(
                            Flex::row()
                                .with_main_axis_size(MainAxisSize::Max)
                                .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                                .with_child(
                                    ConstrainedBox::new(
                                        self.render_project_canvas_palette(appearance, graph),
                                    )
                                    .with_width(232.)
                                    .finish(),
                                )
                                .with_child(
                                    Expanded::new(
                                        1.,
                                        Container::new(
                                            self.render_project_graph_surface(appearance, graph),
                                        )
                                        .with_margin_left(12.)
                                        .finish(),
                                    )
                                    .finish(),
                                )
                                .with_child(
                                    Container::new(
                                        ConstrainedBox::new(
                                            self.render_project_canvas_inspector(appearance, graph),
                                        )
                                        .with_width(300.)
                                        .finish(),
                                    )
                                    .with_margin_left(12.)
                                    .finish(),
                                )
                                .finish(),
                        )
                        .with_padding_left(16.)
                        .with_padding_right(16.)
                        .finish(),
                    )
                    .finish(),
                );

                let ready: ColorU = theme.terminal_colors().normal.green.into();
                root.add_child(
                    Container::new(
                        Flex::row()
                            .with_cross_axis_alignment(CrossAxisAlignment::Center)
                            .with_child(
                                ConstrainedBox::new(
                                    Container::new(Empty::new().finish())
                                        .with_background_color(ready)
                                        .with_corner_radius(CornerRadius::with_all(
                                            Radius::Percentage(50.),
                                        ))
                                        .finish(),
                                )
                                .with_width(8.)
                                .with_height(8.)
                                .finish(),
                            )
                            .with_child(
                                Container::new(self.label_text(
                                    appearance,
                                    "Project status:  Ready".to_string(),
                                    12.,
                                    ready,
                                    false,
                                ))
                                .with_margin_left(8.)
                                .finish(),
                            )
                            .with_child(Expanded::new(1., Empty::new().finish()).finish())
                            .with_child(self.label_text(
                                appearance,
                                format!(
                                    "{} nodes  ·  {} connections  ·  source-only analysis",
                                    graph.nodes.len(),
                                    graph.edges.len()
                                ),
                                11.,
                                theme.disabled_text_color(theme.background()).into(),
                                false,
                            ))
                            .finish(),
                    )
                    .with_uniform_padding(14.)
                    .finish(),
                );
            }
        }

        Container::new(root.finish())
            .with_background(theme.background())
            .finish()
    }

    // ───────────────────── Genesi Probe — rendering ─────────────────────

    pub fn render_api_probe_workspace(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let mut root = Flex::column()
            .with_main_axis_size(MainAxisSize::Max)
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(
                Container::new(self.render_probe_header(appearance))
                    .with_padding_left(16.)
                    .with_padding_right(16.)
                    .with_padding_top(12.)
                    .with_padding_bottom(12.)
                    .finish(),
            );

        match &self.probe_state {
            ProbeState::NoProject => root.add_child(
                Expanded::new(
                    1.,
                    Container::new(self.render_canvas_message(
                        appearance,
                        "No project detected",
                        "Open a project folder or focus a source file, then refresh Probe. You can still type a URL by hand once a project is open.",
                    ))
                    .with_uniform_padding(120.)
                    .finish(),
                )
                .finish(),
            ),
            ProbeState::Loading(path) => root.add_child(
                Expanded::new(
                    1.,
                    Container::new(self.render_canvas_message(
                        appearance,
                        "Surveying project…",
                        &format!(
                            "Reading {} for endpoints and declared ports, and checking which ports are listening. No request is sent while scanning.",
                            path.display()
                        ),
                    ))
                    .with_uniform_padding(120.)
                    .finish(),
                )
                .finish(),
            ),
            ProbeState::Error(error) => root.add_child(
                Expanded::new(
                    1.,
                    Container::new(self.render_canvas_message(appearance, "Survey failed", error))
                        .with_uniform_padding(120.)
                        .finish(),
                )
                .finish(),
            ),
            ProbeState::Ready(survey) => {
                root.add_child(
                    Expanded::new(
                        1.,
                        Container::new(
                            Flex::row()
                                .with_main_axis_size(MainAxisSize::Max)
                                .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                                .with_child(
                                    ConstrainedBox::new(
                                        self.render_probe_sidebar(appearance, survey),
                                    )
                                    .with_width(272.)
                                    .finish(),
                                )
                                .with_child(
                                    Expanded::new(
                                        1.,
                                        Container::new(self.render_probe_console(appearance))
                                            .with_margin_left(12.)
                                            .finish(),
                                    )
                                    .finish(),
                                )
                                .finish(),
                        )
                        .with_padding_left(16.)
                        .with_padding_right(16.)
                        .finish(),
                    )
                    .finish(),
                );
                root.add_child(self.render_probe_status_bar(appearance, survey));
            }
        }

        Container::new(root.finish())
            .with_background(theme.background())
            .finish()
    }

    fn render_probe_header(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let accent: ColorU = theme.terminal_colors().normal.cyan.into();
        let subtitle = match &self.probe_state {
            ProbeState::Ready(survey) => format!(
                "{}  ·  {} endpoints  ·  {} of {} ports open",
                survey.project_name,
                survey.routes.len(),
                survey.open_port_count(),
                survey.ports.len()
            ),
            _ => "Endpoints, ports, and requests — without leaving the editor".to_string(),
        };

        Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_child(
                ConstrainedBox::new(
                    Container::new(
                        CoreIcon::Globe
                            .to_warpui_icon(ThemeFill::Solid(accent))
                            .finish(),
                    )
                    .with_uniform_padding(8.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9.)))
                    .with_background(
                        theme
                            .surface_1()
                            .blend(&ThemeFill::Solid(accent).with_opacity(42)),
                    )
                    .finish(),
                )
                .with_width(34.)
                .with_height(34.)
                .finish(),
            )
            .with_child(
                Container::new(
                    Flex::column()
                        .with_child(self.label_text(
                            appearance,
                            "Genesi Probe".to_string(),
                            18.,
                            theme.main_text_color(theme.background()).into(),
                            false,
                        ))
                        .with_child(self.label_text(
                            appearance,
                            subtitle,
                            11.,
                            theme.disabled_text_color(theme.background()).into(),
                            false,
                        ))
                        .finish(),
                )
                .with_margin_left(12.)
                .finish(),
            )
            .with_child(Expanded::new(1., Empty::new().finish()).finish())
            .with_child(
                Container::new(self.workspace_chip(
                    appearance,
                    "Rescan ports".to_string(),
                    None,
                    WorkspaceAction::RescanGenesiProbePorts,
                    false,
                ))
                .with_margin_left(10.)
                .finish(),
            )
            .with_child(
                Container::new(self.workspace_chip(
                    appearance,
                    "Refresh".to_string(),
                    None,
                    WorkspaceAction::RefreshGenesiProbe,
                    false,
                ))
                .with_margin_left(8.)
                .finish(),
            )
            .with_child(
                Container::new(self.workspace_chip(
                    appearance,
                    "Canvas".to_string(),
                    None,
                    WorkspaceAction::OpenGenesiCanvasTool,
                    false,
                ))
                .with_margin_left(8.)
                .finish(),
            )
            .with_child(
                Container::new(self.workspace_chip(
                    appearance,
                    "Close".to_string(),
                    None,
                    WorkspaceAction::CloseGenesiProbe,
                    false,
                ))
                .with_margin_left(8.)
                .finish(),
            )
            .finish()
    }

    fn render_probe_status_bar(
        &self,
        appearance: &Appearance,
        survey: &ProbeSurvey,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let colors = &theme.terminal_colors().normal;
        let (dot, label): (ColorU, String) = match survey.preferred_port() {
            Some(port) => (colors.green.into(), format!("Target:  {}", port.base_url())),
            None => (
                colors.yellow.into(),
                "No listening port found — start the server, or send to any URL by hand"
                    .to_string(),
            ),
        };
        let stacks = if survey.stacks.is_empty() {
            "source-only analysis".to_string()
        } else {
            survey.stacks.join(" · ")
        };

        Container::new(
            Flex::row()
                .with_cross_axis_alignment(CrossAxisAlignment::Center)
                .with_child(
                    ConstrainedBox::new(
                        Container::new(Empty::new().finish())
                            .with_background_color(dot)
                            .with_corner_radius(CornerRadius::with_all(Radius::Percentage(50.)))
                            .finish(),
                    )
                    .with_width(8.)
                    .with_height(8.)
                    .finish(),
                )
                .with_child(
                    Container::new(self.label_text(appearance, label, 12., dot, false))
                        .with_margin_left(8.)
                        .finish(),
                )
                .with_child(Expanded::new(1., Empty::new().finish()).finish())
                .with_child(self.label_text(
                    appearance,
                    stacks,
                    11.,
                    theme.disabled_text_color(theme.background()).into(),
                    false,
                ))
                .finish(),
        )
        .with_uniform_padding(14.)
        .finish()
    }

    fn render_probe_section_label(
        &self,
        appearance: &Appearance,
        label: String,
        top_margin: f32,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        Container::new(self.label_text(
            appearance,
            label,
            10.,
            theme.disabled_text_color(theme.background()).into(),
            false,
        ))
        .with_margin_top(top_margin)
        .with_margin_bottom(6.)
        .with_padding_left(2.)
        .finish()
    }

    fn render_probe_sidebar(
        &self,
        appearance: &Appearance,
        survey: &ProbeSurvey,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let mut list = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);

        list.add_child(self.render_probe_section_label(
            appearance,
            format!("PORTS  ·  {} OPEN", survey.open_port_count()),
            0.,
        ));
        for port in &survey.ports {
            list.add_child(self.render_probe_port_row(appearance, port));
        }

        list.add_child(self.render_probe_section_label(
            appearance,
            format!("ENDPOINTS  ·  {}", survey.routes.len()),
            16.,
        ));
        if survey.routes.is_empty() {
            list.add_child(
                Container::new(self.label_text(
                    appearance,
                    "No endpoints found in source. Type a URL above and send anyway.".to_string(),
                    11.,
                    muted,
                    true,
                ))
                .with_uniform_padding(8.)
                .finish(),
            );
        }
        for route in &survey.routes {
            list.add_child(self.render_probe_route_row(appearance, route));
        }

        if !self.probe_history.is_empty() {
            list.add_child(
                Flex::row()
                    .with_cross_axis_alignment(CrossAxisAlignment::Center)
                    .with_child(
                        Expanded::new(
                            1.,
                            self.render_probe_section_label(appearance, "HISTORY".to_string(), 16.),
                        )
                        .finish(),
                    )
                    .with_child(self.workspace_chip(
                        appearance,
                        "Clear".to_string(),
                        None,
                        WorkspaceAction::ClearGenesiProbeHistory,
                        false,
                    ))
                    .finish(),
            );
            for (index, entry) in self.probe_history.iter().enumerate() {
                list.add_child(self.render_probe_history_row(appearance, index, entry));
            }
        }

        Container::new(
            ClippedScrollable::vertical(
                self.probe_sidebar_scroll.clone(),
                Container::new(list.finish())
                    .with_uniform_padding(10.)
                    .finish(),
                ScrollbarWidth::Auto,
                theme.disabled_ui_text_color().into(),
                theme.active_ui_text_color().into(),
                Fill::None,
            )
            .finish(),
        )
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(12.)))
        .with_background(theme.surface_1())
        .with_border(Border::all(1.).with_border_fill(theme.outline()))
        .finish()
    }

    fn render_probe_port_row(&self, appearance: &Appearance, port: &ProbePort) -> Box<dyn Element> {
        let theme = appearance.theme();
        let port_number = port.port;
        let accent: ColorU = if port.open {
            theme.terminal_colors().normal.green.into()
        } else {
            theme.disabled_text_color(theme.background()).into()
        };
        let row = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_child(
                ConstrainedBox::new(
                    Container::new(Empty::new().finish())
                        .with_background_color(accent)
                        .with_corner_radius(CornerRadius::with_all(Radius::Percentage(50.)))
                        .finish(),
                )
                .with_width(7.)
                .with_height(7.)
                .finish(),
            )
            .with_child(
                Container::new(
                    Flex::column()
                        .with_cross_axis_alignment(CrossAxisAlignment::Start)
                        .with_child(self.mono_text(
                            appearance,
                            port.port.to_string(),
                            13.,
                            theme.main_text_color(theme.background()).into(),
                            false,
                        ))
                        .with_child(self.label_text(
                            appearance,
                            port.detail(),
                            10.,
                            theme.disabled_text_color(theme.background()).into(),
                            false,
                        ))
                        .finish(),
                )
                .with_margin_left(9.)
                .finish(),
            )
            .finish();

        EventHandler::new(
            Container::new(row)
                .with_horizontal_padding(8.)
                .with_vertical_padding(7.)
                .with_margin_bottom(2.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                .with_background(theme.background())
                .with_border(Border::all(1.).with_border_fill(theme.outline()))
                .finish(),
        )
        .on_left_mouse_down(move |ctx, _, _| {
            ctx.dispatch_typed_action(WorkspaceAction::SelectGenesiProbePort(port_number));
            DispatchEventResult::StopPropagation
        })
        .finish()
    }

    fn render_probe_route_row(
        &self,
        appearance: &Appearance,
        route: &ProbeRoute,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let selected = self.probe_selected_route.as_deref() == Some(route.id.as_str());
        let accent = self.probe_method_color(appearance, &route.method);
        let subtitle = match route.placeholders().as_slice() {
            [] => format!("{}:{}", route.source.display(), route.line),
            parameters => format!("fill in {}", parameters.join(", ")),
        };

        let row = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_child(self.render_probe_method_badge(appearance, &route.method))
            .with_child(
                Expanded::new(
                    1.,
                    Container::new(
                        Flex::column()
                            .with_cross_axis_alignment(CrossAxisAlignment::Start)
                            .with_child(self.mono_text(
                                appearance,
                                route.route.clone(),
                                12.,
                                theme.main_text_color(theme.background()).into(),
                                false,
                            ))
                            .with_child(self.label_text(
                                appearance,
                                subtitle,
                                10.,
                                theme.disabled_text_color(theme.background()).into(),
                                false,
                            ))
                            .finish(),
                    )
                    .with_margin_left(8.)
                    .finish(),
                )
                .finish(),
            )
            .finish();

        let background = if selected {
            theme
                .surface_1()
                .blend(&ThemeFill::Solid(accent).with_opacity(34))
        } else {
            theme.background()
        };
        let border = if selected {
            Border::all(1.).with_border_color(accent)
        } else {
            Border::all(1.).with_border_fill(theme.outline())
        };
        let id = route.id.clone();

        EventHandler::new(
            Container::new(row)
                .with_horizontal_padding(8.)
                .with_vertical_padding(7.)
                .with_margin_bottom(2.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                .with_background(background)
                .with_border(border)
                .finish(),
        )
        .on_left_mouse_down(move |ctx, _, _| {
            ctx.dispatch_typed_action(WorkspaceAction::SelectGenesiProbeRoute(id.clone()));
            DispatchEventResult::StopPropagation
        })
        .finish()
    }

    fn render_probe_history_row(
        &self,
        appearance: &Appearance,
        index: usize,
        entry: &ProbeHistoryEntry,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let colors = &theme.terminal_colors().normal;
        let (status_color, status_label): (ColorU, String) = match entry.status {
            Some(status) if (200..300).contains(&status) => (
                colors.green.into(),
                format!("{status} · {} ms", entry.elapsed_ms),
            ),
            Some(status) if (300..400).contains(&status) => (
                colors.yellow.into(),
                format!("{status} · {} ms", entry.elapsed_ms),
            ),
            Some(status) => (
                colors.red.into(),
                format!("{status} · {} ms", entry.elapsed_ms),
            ),
            None => (colors.red.into(), "no response".to_string()),
        };

        let row = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_child(self.render_probe_method_badge(appearance, &entry.method))
            .with_child(
                Expanded::new(
                    1.,
                    Container::new(
                        Flex::column()
                            .with_cross_axis_alignment(CrossAxisAlignment::Start)
                            .with_child(self.mono_text(
                                appearance,
                                probe_path_of(&entry.url),
                                11.,
                                theme.main_text_color(theme.background()).into(),
                                false,
                            ))
                            .with_child(self.label_text(
                                appearance,
                                status_label,
                                10.,
                                status_color,
                                false,
                            ))
                            .finish(),
                    )
                    .with_margin_left(8.)
                    .finish(),
                )
                .finish(),
            )
            .finish();

        EventHandler::new(
            Container::new(row)
                .with_horizontal_padding(8.)
                .with_vertical_padding(6.)
                .with_margin_bottom(2.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                .with_background(theme.background())
                .with_border(Border::all(1.).with_border_fill(theme.outline()))
                .finish(),
        )
        .on_left_mouse_down(move |ctx, _, _| {
            ctx.dispatch_typed_action(WorkspaceAction::ReplayGenesiProbeHistory(index));
            DispatchEventResult::StopPropagation
        })
        .finish()
    }

    /// Methods are colour-coded the way every API client does it, because the
    /// method is what tells you whether a click is safe.
    fn probe_method_color(&self, appearance: &Appearance, method: &str) -> ColorU {
        let colors = &appearance.theme().terminal_colors().normal;
        match method.trim().to_ascii_uppercase().as_str() {
            "GET" | "HEAD" => colors.green.into(),
            "POST" => colors.yellow.into(),
            "PUT" | "PATCH" => colors.blue.into(),
            "DELETE" => colors.red.into(),
            _ => colors.magenta.into(),
        }
    }

    fn render_probe_method_badge(&self, appearance: &Appearance, method: &str) -> Box<dyn Element> {
        let theme = appearance.theme();
        let accent = self.probe_method_color(appearance, method);
        ConstrainedBox::new(
            Container::new(self.mono_text(appearance, method.to_string(), 9., accent, false))
                .with_horizontal_padding(5.)
                .with_vertical_padding(3.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(5.)))
                .with_background(
                    theme
                        .surface_1()
                        .blend(&ThemeFill::Solid(accent).with_opacity(46)),
                )
                .finish(),
        )
        .with_width(52.)
        .finish()
    }

    fn render_probe_console(&self, appearance: &Appearance) -> Box<dyn Element> {
        Flex::column()
            .with_main_axis_size(MainAxisSize::Max)
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(self.render_probe_request_pane(appearance))
            .with_child(
                Expanded::new(
                    1.,
                    Container::new(self.render_probe_response_pane(appearance))
                        .with_margin_top(12.)
                        .finish(),
                )
                .finish(),
            )
            .finish()
    }

    /// Wrap one of Probe's editors in a surface. The editor draws no frame of
    /// its own, so the container is what makes it read as a field.
    fn render_probe_input(
        &self,
        appearance: &Appearance,
        field: &ViewHandle<EditorView>,
    ) -> Box<dyn Element> {
        appearance
            .ui_builder()
            .text_input(field.clone())
            .with_style(UiComponentStyles {
                background: Some(Fill::Solid(ColorU::transparent_black())),
                border_color: Some(Fill::Solid(ColorU::transparent_black())),
                padding: Some(Coords::uniform(8.)),
                ..Default::default()
            })
            .build()
            .finish()
    }

    fn render_probe_request_pane(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let sending = matches!(self.probe_response, ProbeResponseState::InFlight);

        let mut methods = Flex::row().with_cross_axis_alignment(CrossAxisAlignment::Center);
        for method in PROBE_METHODS {
            methods.add_child(self.workspace_chip(
                appearance,
                (*method).to_string(),
                None,
                WorkspaceAction::SetGenesiProbeMethod((*method).to_string()),
                self.probe_method.eq_ignore_ascii_case(method),
            ));
        }

        let url_row = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_child(
                Expanded::new(
                    1.,
                    Container::new(self.render_probe_input(appearance, &self.probe_url))
                        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                        .with_background(theme.background())
                        .with_border(Border::all(1.).with_border_fill(theme.outline()))
                        .finish(),
                )
                .finish(),
            )
            .with_child(
                Container::new(self.workspace_chip(
                    appearance,
                    if sending {
                        "Sending…".to_string()
                    } else {
                        "Send".to_string()
                    },
                    None,
                    WorkspaceAction::SendGenesiProbeRequest,
                    !sending,
                ))
                .with_margin_left(10.)
                .finish(),
            )
            .finish();

        // The body pane is hidden for methods that do not carry one, so a GET
        // does not offer a field whose contents would be silently dropped.
        let takes_body = method_takes_body(&self.probe_method);
        let showing_headers = self.probe_show_request_headers || !takes_body;
        let mut tabs = Flex::row().with_cross_axis_alignment(CrossAxisAlignment::Center);
        if takes_body {
            tabs.add_child(self.workspace_chip(
                appearance,
                "Body".to_string(),
                None,
                WorkspaceAction::ShowGenesiProbeRequestHeaders(false),
                !showing_headers,
            ));
        }
        tabs.add_child(self.workspace_chip(
            appearance,
            "Headers".to_string(),
            None,
            WorkspaceAction::ShowGenesiProbeRequestHeaders(true),
            showing_headers,
        ));
        if !takes_body {
            tabs.add_child(self.label_text(
                appearance,
                format!("{} sends no body", self.probe_method.to_ascii_uppercase()),
                10.,
                theme.disabled_text_color(theme.background()).into(),
                false,
            ));
        }

        let field = if showing_headers {
            &self.probe_headers_input
        } else {
            &self.probe_body_input
        };

        Container::new(
            Flex::column()
                .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                .with_child(methods.finish())
                .with_child(Container::new(url_row).with_margin_top(10.).finish())
                .with_child(Container::new(tabs.finish()).with_margin_top(12.).finish())
                .with_child(
                    Container::new(
                        ConstrainedBox::new(self.render_probe_input(appearance, field))
                            .with_height(132.)
                            .finish(),
                    )
                    .with_margin_top(8.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                    .with_background(theme.background())
                    .with_border(Border::all(1.).with_border_fill(theme.outline()))
                    .finish(),
                )
                .finish(),
        )
        .with_uniform_padding(14.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(12.)))
        .with_background(theme.surface_1())
        .with_border(Border::all(1.).with_border_fill(theme.outline()))
        .finish()
    }

    fn render_probe_response_pane(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let colors = &theme.terminal_colors().normal;
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();

        let mut header = Flex::row().with_cross_axis_alignment(CrossAxisAlignment::Center);
        let body: Box<dyn Element> = match &self.probe_response {
            ProbeResponseState::Idle => {
                header.add_child(self.label_text(
                    appearance,
                    "Response".to_string(),
                    12.,
                    muted,
                    false,
                ));
                self.label_text(
                    appearance,
                    "Pick an endpoint on the left, or type a URL, then press Send. Enter in the URL field sends too."
                        .to_string(),
                    12.,
                    muted,
                    true,
                )
            }
            ProbeResponseState::InFlight => {
                header.add_child(self.label_text(
                    appearance,
                    "Waiting for the server…".to_string(),
                    12.,
                    colors.yellow.into(),
                    false,
                ));
                self.label_text(appearance, String::new(), 12., muted, false)
            }
            ProbeResponseState::Failed(error) => {
                header.add_child(self.label_text(
                    appearance,
                    "Request failed".to_string(),
                    12.,
                    colors.red.into(),
                    false,
                ));
                header.add_child(Expanded::new(1., Empty::new().finish()).finish());
                header.add_child(self.workspace_chip(
                    appearance,
                    "Copy".to_string(),
                    None,
                    WorkspaceAction::CopyGenesiProbeResponse,
                    false,
                ));
                self.mono_text(
                    appearance,
                    error.clone(),
                    MONO_FONT_SIZE,
                    colors.red.into(),
                    true,
                )
            }
            ProbeResponseState::Ready(response) => {
                let status_color: ColorU = if response.is_success() {
                    colors.green.into()
                } else if response.status >= 500 {
                    colors.red.into()
                } else {
                    colors.yellow.into()
                };
                header.add_child(
                    Container::new(self.mono_text(
                        appearance,
                        response.status_line(),
                        12.,
                        status_color,
                        false,
                    ))
                    .with_horizontal_padding(8.)
                    .with_vertical_padding(4.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
                    .with_background(
                        theme
                            .surface_1()
                            .blend(&ThemeFill::Solid(status_color).with_opacity(46)),
                    )
                    .finish(),
                );
                let mut summary =
                    format!("{} ms  ·  {}", response.elapsed_ms, response.size_label());
                if response.truncated {
                    summary.push_str("  ·  body truncated for display");
                }
                header.add_child(
                    Container::new(self.label_text(appearance, summary, 11., muted, false))
                        .with_margin_left(10.)
                        .finish(),
                );
                header.add_child(Expanded::new(1., Empty::new().finish()).finish());
                header.add_child(self.workspace_chip(
                    appearance,
                    "Body".to_string(),
                    None,
                    WorkspaceAction::ShowGenesiProbeResponseHeaders(false),
                    !self.probe_show_response_headers,
                ));
                header.add_child(self.workspace_chip(
                    appearance,
                    format!("Headers ({})", response.headers.len()),
                    None,
                    WorkspaceAction::ShowGenesiProbeResponseHeaders(true),
                    self.probe_show_response_headers,
                ));
                header.add_child(self.workspace_chip(
                    appearance,
                    "Copy".to_string(),
                    None,
                    WorkspaceAction::CopyGenesiProbeResponse,
                    false,
                ));

                let text = if self.probe_show_response_headers {
                    response
                        .headers
                        .iter()
                        .map(|(key, value)| format!("{key}: {value}"))
                        .collect::<Vec<_>>()
                        .join("\n")
                } else if response.body.trim().is_empty() {
                    "<empty body>".to_string()
                } else {
                    response.body.clone()
                };
                self.mono_text(
                    appearance,
                    text,
                    MONO_FONT_SIZE,
                    theme.main_text_color(theme.background()).into(),
                    true,
                )
            }
        };

        Container::new(
            Flex::column()
                .with_main_axis_size(MainAxisSize::Max)
                .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                .with_child(header.finish())
                .with_child(
                    Expanded::new(
                        1.,
                        Container::new(
                            ClippedScrollable::vertical(
                                self.probe_response_scroll.clone(),
                                body,
                                ScrollbarWidth::Auto,
                                theme.disabled_ui_text_color().into(),
                                theme.active_ui_text_color().into(),
                                Fill::None,
                            )
                            .finish(),
                        )
                        .with_margin_top(10.)
                        .finish(),
                    )
                    .finish(),
                )
                .finish(),
        )
        .with_uniform_padding(14.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(12.)))
        .with_background(theme.surface_1())
        .with_border(Border::all(1.).with_border_fill(theme.outline()))
        .finish()
    }

    fn canvas_node_accent(&self, appearance: &Appearance, kind: CanvasNodeKind) -> ColorU {
        let colors = &appearance.theme().terminal_colors().normal;
        match kind {
            CanvasNodeKind::Page => colors.blue.into(),
            CanvasNodeKind::Router => colors.magenta.into(),
            CanvasNodeKind::Endpoint => colors.green.into(),
        }
    }

    fn render_canvas_node(
        &self,
        appearance: &Appearance,
        node: &CanvasNode,
        selected: bool,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let accent = self.canvas_node_accent(appearance, node.kind);
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let subtitle = match node.kind {
            CanvasNodeKind::Page => node
                .estimated_load_ms
                .map(|load| format!("{}  ·  ~{load} ms estimated load", node.kind.label()))
                .unwrap_or_else(|| node.kind.label().to_string()),
            CanvasNodeKind::Endpoint => format!(
                "{}  ·  {}",
                node.method.as_deref().unwrap_or("ANY"),
                node.kind.label()
            ),
            CanvasNodeKind::Router => "Router / controller".to_string(),
        };
        let source = format!("{}:{}", node.source.display(), node.line);
        let action = WorkspaceAction::SelectGenesiCanvasNode(node.id.clone());

        EventHandler::new(
            Container::new(
                Flex::column()
                    .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                    .with_child(
                        Container::new(
                            ConstrainedBox::new(Empty::new().finish())
                                .with_height(3.)
                                .finish(),
                        )
                        .with_background_color(accent)
                        .finish(),
                    )
                    .with_child(
                        Container::new(self.label_text(
                            appearance,
                            node.title.clone(),
                            TITLE_FONT_SIZE,
                            theme.main_text_color(theme.background()).into(),
                            false,
                        ))
                        .with_margin_top(9.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(self.label_text(
                            appearance,
                            subtitle,
                            CHIP_FONT_SIZE,
                            accent,
                            false,
                        ))
                        .with_margin_top(4.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(self.label_text(
                            appearance,
                            source,
                            CHIP_FONT_SIZE,
                            muted,
                            true,
                        ))
                        .with_margin_top(5.)
                        .finish(),
                    )
                    .finish(),
            )
            .with_padding_left(12.)
            .with_padding_right(12.)
            .with_padding_bottom(11.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9.)))
            .with_border(Border::all(1.).with_border_fill(if selected {
                theme.accent()
            } else {
                theme.outline()
            }))
            .with_background(theme.surface_1())
            .finish(),
        )
        .on_left_mouse_down(move |ctx, _, _| {
            ctx.dispatch_typed_action(action.clone());
            DispatchEventResult::StopPropagation
        })
        .finish()
    }

    fn render_canvas_edge(
        &self,
        appearance: &Appearance,
        graph: &ProjectCanvasGraph,
        edge: &super::project_canvas::CanvasEdge,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let from = graph
            .node(&edge.from)
            .map(|node| node.title.as_str())
            .unwrap_or("Unknown");
        let to = graph
            .node(&edge.to)
            .map(|node| node.title.as_str())
            .unwrap_or("Unknown");
        let source = format!("{}:{}", edge.source.display(), edge.line);
        let accent: ColorU = match edge.kind {
            CanvasEdgeKind::PageLink => theme.terminal_colors().normal.blue.into(),
            CanvasEdgeKind::ApiCall | CanvasEdgeKind::InternalCall => {
                theme.terminal_colors().normal.green.into()
            }
            CanvasEdgeKind::Defines => theme.terminal_colors().normal.magenta.into(),
        };
        let action = WorkspaceAction::SelectGenesiCanvasNode(edge.to.clone());

        EventHandler::new(
            Container::new(
                Flex::column()
                    .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                    .with_child(self.label_text(
                        appearance,
                        format!("{from}  →  {to}"),
                        BODY_FONT_SIZE,
                        theme.main_text_color(theme.background()).into(),
                        false,
                    ))
                    .with_child(
                        Container::new(self.label_text(
                            appearance,
                            format!("{}  ·  {}  ·  {source}", edge.kind.label(), edge.label),
                            CHIP_FONT_SIZE,
                            muted,
                            true,
                        ))
                        .with_margin_top(4.)
                        .finish(),
                    )
                    .finish(),
            )
            .with_uniform_padding(10.)
            .with_border(Border::left(2.).with_border_color(accent))
            .with_background(theme.surface_1())
            .finish(),
        )
        .on_left_mouse_down(move |ctx, _, _| {
            ctx.dispatch_typed_action(action.clone());
            DispatchEventResult::StopPropagation
        })
        .finish()
    }

    fn render_canvas_details(
        &self,
        appearance: &Appearance,
        graph: &ProjectCanvasGraph,
        node: &CanvasNode,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let accent = self.canvas_node_accent(appearance, node.kind);
        let mut details = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(self.label_text(
                appearance,
                "Selected node".to_string(),
                CHIP_FONT_SIZE,
                accent,
                false,
            ))
            .with_child(
                Container::new(self.label_text(
                    appearance,
                    node.title.clone(),
                    TITLE_FONT_SIZE,
                    theme.main_text_color(theme.background()).into(),
                    false,
                ))
                .with_margin_top(5.)
                .finish(),
            )
            .with_child(
                Container::new(self.label_text(
                    appearance,
                    format!(
                        "{}:{}  ·  {}",
                        node.source.display(),
                        node.line,
                        node.kind.label()
                    ),
                    CHIP_FONT_SIZE,
                    muted,
                    true,
                ))
                .with_margin_top(5.)
                .finish(),
            );

        if let Some(body) = &node.request_body {
            details.add_child(
                Container::new(self.label_text(
                    appearance,
                    format!("Request body\n{body}"),
                    BODY_FONT_SIZE,
                    theme.main_text_color(theme.background()).into(),
                    true,
                ))
                .with_margin_top(12.)
                .finish(),
            );
        }
        if !node.dependencies.is_empty() {
            details.add_child(
                Container::new(self.label_text(
                    appearance,
                    format!("Dependencies\n{}", node.dependencies.join("  ·  ")),
                    BODY_FONT_SIZE,
                    theme.main_text_color(theme.background()).into(),
                    true,
                ))
                .with_margin_top(12.)
                .finish(),
            );
        }
        let connections = graph
            .edges
            .iter()
            .filter(|edge| edge.from == node.id || edge.to == node.id)
            .count();
        details.add_child(
            Container::new(self.label_text(
                appearance,
                format!(
                    "{connections} graph connection{}",
                    if connections == 1 { "" } else { "s" }
                ),
                CHIP_FONT_SIZE,
                muted,
                false,
            ))
            .with_margin_top(12.)
            .finish(),
        );

        Container::new(details.finish())
            .with_uniform_padding(14.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9.)))
            .with_border(Border::all(1.).with_border_fill(theme.accent()))
            .with_background(theme.surface_1())
            .finish()
    }

    fn render_canvas_message(
        &self,
        appearance: &Appearance,
        title: &str,
        body: &str,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        Container::new(
            Flex::column()
                .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                .with_child(self.label_text(
                    appearance,
                    title.to_string(),
                    TITLE_FONT_SIZE,
                    theme.main_text_color(theme.background()).into(),
                    false,
                ))
                .with_child(
                    Container::new(self.label_text(
                        appearance,
                        body.to_string(),
                        BODY_FONT_SIZE,
                        theme.disabled_text_color(theme.background()).into(),
                        true,
                    ))
                    .with_margin_top(7.)
                    .finish(),
                )
                .finish(),
        )
        .with_uniform_padding(14.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9.)))
        .with_border(Border::all(1.).with_border_fill(theme.outline()))
        .with_background(theme.surface_1())
        .finish()
    }

    fn render_project_canvas(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let mut root = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);

        match &self.project_canvas_state {
            ProjectCanvasState::NoProject => {
                root.add_child(self.render_canvas_message(
                    appearance,
                    "No project detected",
                    "Open a project folder or focus a source file, then open Project Canvas again.",
                ));
            }
            ProjectCanvasState::Loading(path) => {
                root.add_child(self.render_canvas_message(
                    appearance,
                    "Analyzing project…",
                    &format!(
                        "Scanning {} for pages, links, routers, and endpoints. Project code is never executed.",
                        path.display()
                    ),
                ));
            }
            ProjectCanvasState::Error(error) => {
                root.add_child(self.render_canvas_message(
                    appearance,
                    "Project analysis failed",
                    error,
                ));
            }
            ProjectCanvasState::Ready(graph) if graph.kind == ProjectKind::Unknown => {
                root.add_child(self.render_canvas_message(
                    appearance,
                    "No recognized project structure",
                    &format!(
                        "Scanned {} source files in {} but found no supported pages or endpoints.",
                        graph.files_scanned,
                        graph.root.display()
                    ),
                ));
            }
            ProjectCanvasState::Ready(graph) => {
                let stacks = if graph.stacks.is_empty() {
                    "Framework inferred from source".to_string()
                } else {
                    graph.stacks.join("  ·  ")
                };
                root.add_child(
                    Flex::row()
                        .with_main_axis_alignment(MainAxisAlignment::SpaceBetween)
                        .with_cross_axis_alignment(CrossAxisAlignment::Center)
                        .with_child(
                            Flex::column()
                                .with_child(self.label_text(
                                    appearance,
                                    graph.project_name.clone(),
                                    TITLE_FONT_SIZE,
                                    theme.main_text_color(theme.background()).into(),
                                    false,
                                ))
                                .with_child(
                                    Container::new(self.label_text(
                                        appearance,
                                        format!(
                                            "{}  ·  {} pages  ·  {} endpoints",
                                            graph.kind.label(),
                                            graph.page_count(),
                                            graph.endpoint_count()
                                        ),
                                        CHIP_FONT_SIZE,
                                        theme.disabled_text_color(theme.background()).into(),
                                        false,
                                    ))
                                    .with_margin_top(3.)
                                    .finish(),
                                )
                                .finish(),
                        )
                        .with_child(self.workspace_chip(
                            appearance,
                            "Refresh".to_string(),
                            None,
                            WorkspaceAction::RefreshGenesiCanvas,
                            false,
                        ))
                        .finish(),
                );
                root.add_child(
                    Container::new(self.label_text(
                        appearance,
                        stacks,
                        CHIP_FONT_SIZE,
                        theme.accent().into_solid(),
                        true,
                    ))
                    .with_margin_top(6.)
                    .finish(),
                );

                if let Some(selected) = self
                    .selected_canvas_node
                    .as_deref()
                    .and_then(|id| graph.node(id))
                {
                    root.add_child(
                        Container::new(self.render_canvas_details(appearance, graph, selected))
                            .with_margin_top(14.)
                            .finish(),
                    );
                }

                for (kind, heading) in [
                    (CanvasNodeKind::Page, "Pages"),
                    (CanvasNodeKind::Router, "Backend routers"),
                    (CanvasNodeKind::Endpoint, "Endpoints"),
                ] {
                    let section_nodes = graph.nodes.iter().filter(|node| node.kind == kind);
                    let mut section = Flex::column()
                        .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                        .with_child(self.label_text(
                            appearance,
                            heading.to_string(),
                            BODY_FONT_SIZE,
                            theme.main_text_color(theme.background()).into(),
                            false,
                        ));
                    let mut count = 0;
                    for node in section_nodes {
                        count += 1;
                        section.add_child(
                            Container::new(self.render_canvas_node(
                                appearance,
                                node,
                                self.selected_canvas_node.as_deref() == Some(node.id.as_str()),
                            ))
                            .with_margin_top(8.)
                            .finish(),
                        );
                    }
                    if count > 0 {
                        root.add_child(
                            Container::new(section.finish())
                                .with_margin_top(16.)
                                .finish(),
                        );
                    }
                }

                if !graph.edges.is_empty() {
                    let mut connections = Flex::column()
                        .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                        .with_child(self.label_text(
                            appearance,
                            "Connections".to_string(),
                            BODY_FONT_SIZE,
                            theme.main_text_color(theme.background()).into(),
                            false,
                        ));
                    for edge in graph.edges.iter().take(120) {
                        connections.add_child(
                            Container::new(self.render_canvas_edge(appearance, graph, edge))
                                .with_margin_top(7.)
                                .finish(),
                        );
                    }
                    root.add_child(
                        Container::new(connections.finish())
                            .with_margin_top(16.)
                            .finish(),
                    );
                }
            }
        }

        Container::new(root.finish())
            .with_uniform_padding(16.)
            .finish()
    }

    pub fn current_chat_title(&self) -> String {
        Self::chat_title_from_entries(&self.messages)
    }

    pub fn start_new_chat(&mut self, ctx: &mut ViewContext<Self>) {
        self.stop_turn(ctx);
        self.persist_mempalace(ctx, false);
        if self.messages.is_empty() {
            self.reset_current_chat();
        } else {
            self.active_chat_id = Self::new_chat_id();
            self.reset_current_chat();
        }
        self.persist_mempalace(ctx, true);
        ctx.notify();
    }

    pub fn set_vibe_mode(&mut self, enabled: bool, ctx: &mut ViewContext<Self>) {
        self.vibe_mode = enabled;
        ctx.notify();
    }

    pub fn select_review_file(&mut self, path: &str, ctx: &mut ViewContext<Self>) {
        if self.pending_edits.iter().any(|edit| edit.path == path) {
            self.active_side_tool = GenesiSideTool::Review;
            self.selected_review_path = Some(path.to_string());
            self.review_expanded = true;
            ctx.notify();
        }
    }

    pub fn open_review_tool(&mut self, ctx: &mut ViewContext<Self>) {
        self.active_side_tool = GenesiSideTool::Review;
        ctx.notify();
    }

    fn auto_arranged_project_canvas_positions(
        graph: &ProjectCanvasGraph,
    ) -> HashMap<String, Vector2F> {
        const ROWS_PER_COLUMN: usize = 5;
        const COLUMN_GAP: f32 = 270.;
        const ROW_GAP: f32 = 170.;
        const START_X: f32 = 70.;
        const START_Y: f32 = 80.;

        let page_count = graph
            .nodes
            .iter()
            .filter(|node| node.kind == CanvasNodeKind::Page)
            .count();
        let router_count = graph
            .nodes
            .iter()
            .filter(|node| node.kind == CanvasNodeKind::Router)
            .count();
        let columns_for = |count: usize| count.max(1).div_ceil(ROWS_PER_COLUMN) as f32;
        let page_base = START_X;
        let router_base = page_base
            + if page_count > 0 {
                columns_for(page_count) * COLUMN_GAP
            } else {
                0.
            };
        let endpoint_base = router_base
            + if router_count > 0 {
                columns_for(router_count) * COLUMN_GAP
            } else if page_count > 0 {
                COLUMN_GAP
            } else {
                0.
            };
        let mut indexes = HashMap::<CanvasNodeKind, usize>::new();
        let mut positions = HashMap::new();
        for node in &graph.nodes {
            let index = indexes.entry(node.kind).or_default();
            let base_x = match node.kind {
                CanvasNodeKind::Page => page_base,
                CanvasNodeKind::Router => router_base,
                CanvasNodeKind::Endpoint => endpoint_base,
            };
            let column = *index / ROWS_PER_COLUMN;
            let row = *index % ROWS_PER_COLUMN;
            positions.insert(
                node.id.clone(),
                vec2f(
                    base_x + column as f32 * COLUMN_GAP,
                    START_Y + row as f32 * ROW_GAP,
                ),
            );
            *index += 1;
        }
        positions
    }

    pub fn auto_arrange_project_canvas(&mut self, ctx: &mut ViewContext<Self>) {
        if let ProjectCanvasState::Ready(graph) = &self.project_canvas_state {
            self.project_canvas_positions =
                Arc::new(Self::auto_arranged_project_canvas_positions(graph));
            self.project_canvas_pan = vec2f(42., 42.);
            self.project_canvas_drag = None;
            self.reset_project_canvas_drag_sampling();
            ctx.notify();
        }
    }

    pub fn fit_project_canvas(&mut self, ctx: &mut ViewContext<Self>) {
        let node_count = match &self.project_canvas_state {
            ProjectCanvasState::Ready(graph) => graph.nodes.len(),
            _ => 0,
        };
        self.project_canvas_zoom = if node_count > 24 {
            0.6
        } else if node_count > 14 {
            0.75
        } else if node_count > 8 {
            0.85
        } else {
            1.
        };
        self.project_canvas_pan = vec2f(42., 42.);
        self.project_canvas_drag = None;
        self.reset_project_canvas_drag_sampling();
        ctx.notify();
    }

    pub fn zoom_project_canvas(&mut self, delta: f32, ctx: &mut ViewContext<Self>) {
        self.project_canvas_zoom = (self.project_canvas_zoom + delta).clamp(0.45, 1.6);
        ctx.notify();
    }

    pub fn pan_project_canvas(&mut self, delta: Vector2F, ctx: &mut ViewContext<Self>) {
        self.project_canvas_pan += delta;
        ctx.notify();
    }

    fn reset_project_canvas_drag_sampling(&mut self) {
        self.project_canvas_last_drag_frame = None;
        self.project_canvas_last_drag_pointer = None;
        self.project_canvas_pending_drag_pointer = None;
    }

    fn apply_project_canvas_drag_pointer(&mut self, pointer: Vector2F) -> bool {
        match &self.project_canvas_drag {
            Some(ProjectCanvasDragState::Pan {
                pointer: start,
                pan,
            }) => {
                self.project_canvas_pan = *pan + pointer - *start;
            }
            Some(ProjectCanvasDragState::Node {
                id,
                pointer: start,
                position,
            }) => {
                let delta = (pointer - *start) / self.project_canvas_zoom;
                let next = *position + delta;
                Arc::make_mut(&mut self.project_canvas_positions)
                    .insert(id.clone(), vec2f(next.x().max(0.), next.y().max(0.)));
            }
            None => return false,
        }
        true
    }

    pub fn begin_project_canvas_drag(
        &mut self,
        node_id: Option<String>,
        pointer: Vector2F,
        ctx: &mut ViewContext<Self>,
    ) {
        self.project_canvas_drag = if let Some(id) = node_id {
            let position = self
                .project_canvas_positions
                .get(&id)
                .copied()
                .unwrap_or_default();
            self.selected_canvas_node = Some(id.clone());
            Some(ProjectCanvasDragState::Node {
                id,
                pointer,
                position,
            })
        } else {
            Some(ProjectCanvasDragState::Pan {
                pointer,
                pan: self.project_canvas_pan,
            })
        };
        self.project_canvas_last_drag_frame = Some(Instant::now());
        self.project_canvas_last_drag_pointer = Some(pointer);
        self.project_canvas_pending_drag_pointer = Some(pointer);
        ctx.notify();
    }

    pub fn update_project_canvas_drag(&mut self, pointer: Vector2F, ctx: &mut ViewContext<Self>) {
        if self.project_canvas_drag.is_none() {
            return;
        }

        // Apply every move. The previous throttle stored the pointer as "pending"
        // and returned, but nothing ever came back to flush it — the only flush
        // was on mouse-up. Since pointer events arrive far more often than the
        // 16ms budget, most of a gesture was DISCARDED rather than deferred, so
        // the canvas appeared to freeze mid-drag and jump at the end. The frame
        // budget that protects the render already lives in the canvas view
        // (it draws fewer nodes while dragging), which is the right place for it:
        // dropping input is not throttling, it is losing the gesture.
        let now = Instant::now();
        self.project_canvas_pending_drag_pointer = Some(pointer);
        if self.apply_project_canvas_drag_pointer(pointer) {
            self.project_canvas_last_drag_frame = Some(now);
            self.project_canvas_last_drag_pointer = Some(pointer);
            ctx.notify();
        }
    }

    pub fn end_project_canvas_drag(&mut self, ctx: &mut ViewContext<Self>) {
        if let Some(pointer) = self.project_canvas_pending_drag_pointer.take() {
            self.apply_project_canvas_drag_pointer(pointer);
        }
        let ended = self.project_canvas_drag.take().is_some();
        self.reset_project_canvas_drag_sampling();
        if ended {
            ctx.notify();
        }
    }

    // ──────────────────────────── Genesi Probe ────────────────────────────
    //
    // The API client. Discovery is shared with the Canvas (same walk, same
    // routes); what Probe adds is the port survey and the round trip.

    /// One of Probe's three input fields.
    ///
    /// These are plain [`EditorView`]s rather than [`SubmittableTextInput`]s
    /// because that component clears its buffer on submit — correct for a chat
    /// compose box, wrong for a URL you send, adjust, and send again.
    fn probe_editor(
        ctx: &mut ViewContext<Self>,
        single_line: bool,
        placeholder: &'static str,
    ) -> ViewHandle<EditorView> {
        let editor = ctx.add_typed_action_view(|ctx| {
            let appearance = Appearance::as_ref(ctx);
            let options = EditorOptions {
                // The URL grows to its content; the body and header panes get a
                // fixed box and scroll inside it, the way a code editor does.
                autogrow: single_line,
                soft_wrap: !single_line,
                single_line,
                text: TextOptions::ui_font_size(appearance),
                ..Default::default()
            };
            EditorView::new(options, ctx)
        });
        editor.update(ctx, |editor, ctx| {
            editor.set_placeholder_text(placeholder, ctx);
        });
        editor
    }

    fn probe_field_text(
        &self,
        field: &ViewHandle<EditorView>,
        ctx: &mut ViewContext<Self>,
    ) -> String {
        field.read(ctx, |editor, ctx| editor.buffer_text(ctx))
    }

    fn set_probe_field(
        &self,
        field: &ViewHandle<EditorView>,
        value: &str,
        ctx: &mut ViewContext<Self>,
    ) {
        field.update(ctx, |editor, ctx| editor.set_buffer_text(value, ctx));
    }

    pub fn open_api_probe(&mut self, project_root: Option<PathBuf>, ctx: &mut ViewContext<Self>) {
        // Reopening Probe on the project it already surveyed must not throw the
        // survey away — the ports it found are the expensive part.
        let already_surveyed = match (&self.probe_state, &project_root) {
            (ProbeState::Ready(survey), Some(root)) => &survey.root == root,
            _ => false,
        };
        if already_surveyed {
            ctx.notify();
            return;
        }
        self.refresh_api_probe(project_root, ctx);
    }

    pub fn refresh_api_probe(
        &mut self,
        project_root: Option<PathBuf>,
        ctx: &mut ViewContext<Self>,
    ) {
        self.probe_generation = self.probe_generation.wrapping_add(1);
        let generation = self.probe_generation;
        let Some(root) = project_root else {
            self.probe_state = ProbeState::NoProject;
            self.probe_selected_route = None;
            ctx.notify();
            return;
        };

        self.agent_root = Some(root.clone());
        self.probe_state = ProbeState::Loading(root.clone());
        ctx.notify();

        // Hand the Canvas its graph back when it already analysed this exact
        // project: the route half of the survey IS that walk, and repeating it
        // is the difference between Probe opening instantly and stalling.
        let graph = match &self.project_canvas_state {
            ProjectCanvasState::Ready(graph) if graph.root == root => Some(graph.clone()),
            _ => None,
        };

        ctx.spawn(
            async move { tokio::task::spawn_blocking(move || survey_project(&root, graph)).await },
            move |me, result, ctx| {
                if me.probe_generation != generation {
                    return;
                }
                match result {
                    Ok(survey) => me.apply_probe_survey(Arc::new(survey), ctx),
                    Err(error) => {
                        me.probe_state = ProbeState::Error(format!(
                            "The background scanner stopped unexpectedly: {error}"
                        ));
                        me.probe_selected_route = None;
                    }
                }
                ctx.notify();
            },
        );
    }

    /// Write a URL into the field and remember exactly what was written, so a
    /// later survey can tell its own seed apart from something the user typed.
    fn seed_probe_url(&mut self, url: &str, ctx: &mut ViewContext<Self>) {
        self.set_probe_field(&self.probe_url, url, ctx);
        self.probe_seeded_url = Some(url.to_string());
    }

    /// Whether the URL field still holds exactly what Probe last put there.
    fn probe_url_untouched(&self, ctx: &mut ViewContext<Self>) -> bool {
        let current = self.probe_field_text(&self.probe_url, ctx);
        if current.trim().is_empty() {
            return true;
        }
        self.probe_seeded_url.as_deref() == Some(current.as_str())
    }

    fn apply_probe_survey(&mut self, survey: Arc<ProbeSurvey>, ctx: &mut ViewContext<Self>) {
        let base = survey.base_url();

        // A route the user clicked on the Canvas before the survey finished.
        // It could not become a URL then — the port was not known yet — so it
        // waited here for the real one instead of being built against a guess.
        if let Some((method, route, body_hint)) = self.probe_pending_route.take() {
            let takes_body = method_takes_body(&method);
            self.probe_method = method;
            self.seed_probe_url(&build_url(&base, &route), ctx);
            if takes_body {
                self.seed_probe_body(body_hint.as_deref(), ctx);
            }
            self.probe_state = ProbeState::Ready(survey);
            return;
        }

        // Otherwise re-point whatever Probe seeded earlier at the port the
        // survey actually found. Only its own seed is rewritten: a URL the user
        // typed is theirs, and a rescan landing mid-edit must not clobber it.
        if self.probe_url_untouched(ctx) {
            let url = match survey.routes.first() {
                Some(route) => {
                    self.probe_method = route.method.clone();
                    self.probe_selected_route = Some(route.id.clone());
                    build_url(&base, &route.route)
                }
                None => base,
            };
            self.seed_probe_url(&url, ctx);
        }
        self.probe_state = ProbeState::Ready(survey);
    }

    /// Re-check ports without re-walking the project. This is the button people
    /// press, because the thing that changes between two looks is whether the
    /// dev server came up — not the routes.
    pub fn rescan_probe_ports(&mut self, ctx: &mut ViewContext<Self>) {
        let ProbeState::Ready(survey) = &self.probe_state else {
            return;
        };
        let survey = survey.clone();
        let root = survey.root.clone();
        let generation = self.probe_generation;
        ctx.spawn(
            async move { tokio::task::spawn_blocking(move || scan_ports(&root)).await },
            move |me, result, ctx| {
                if me.probe_generation != generation {
                    return;
                }
                if let Ok(ports) = result {
                    let mut refreshed = (*survey).clone();
                    refreshed.ports = ports;
                    me.probe_state = ProbeState::Ready(Arc::new(refreshed));
                }
                ctx.notify();
            },
        );
    }

    pub fn select_probe_route(&mut self, id: &str, ctx: &mut ViewContext<Self>) {
        let ProbeState::Ready(survey) = &self.probe_state else {
            return;
        };
        let Some(route) = survey.route(id) else {
            return;
        };
        let method = route.method.clone();
        let url = build_url(&survey.base_url(), &route.route);
        let body_hint = route.body_hint.clone();

        self.probe_selected_route = Some(id.to_string());
        self.probe_method = method.clone();
        self.seed_probe_url(&url, ctx);
        if method_takes_body(&method) {
            self.seed_probe_body(body_hint.as_deref(), ctx);
            self.probe_show_request_headers = false;
        }
        ctx.notify();
    }

    /// Lay out the fields the handler appears to read, so a POST is one edit
    /// away from sendable instead of an empty box. Never overwrites a body the
    /// user already wrote.
    fn seed_probe_body(&mut self, hint: Option<&str>, ctx: &mut ViewContext<Self>) {
        let Some(template) = hint.and_then(probe_body_template) else {
            return;
        };
        let current = self.probe_field_text(&self.probe_body_input, ctx);
        if current.trim().is_empty() {
            self.set_probe_field(&self.probe_body_input, &template, ctx);
        }
    }

    /// Retarget the current request at another port, keeping the path — that is
    /// the whole point of having the port list next to the request.
    pub fn select_probe_port(&mut self, port: u16, ctx: &mut ViewContext<Self>) {
        let current = self.probe_field_text(&self.probe_url, ctx);
        let url = build_url(
            &format!("http://127.0.0.1:{port}"),
            &probe_path_of(&current),
        );
        self.seed_probe_url(&url, ctx);
        ctx.notify();
    }

    pub fn set_probe_method(&mut self, method: &str, ctx: &mut ViewContext<Self>) {
        self.probe_method = method.to_string();
        ctx.notify();
    }

    pub fn show_probe_request_headers(&mut self, show: bool, ctx: &mut ViewContext<Self>) {
        self.probe_show_request_headers = show;
        ctx.notify();
    }

    pub fn show_probe_response_headers(&mut self, show: bool, ctx: &mut ViewContext<Self>) {
        self.probe_show_response_headers = show;
        ctx.notify();
    }

    pub fn send_probe_request(&mut self, ctx: &mut ViewContext<Self>) {
        if matches!(self.probe_response, ProbeResponseState::InFlight) {
            return;
        }
        let url = self.probe_field_text(&self.probe_url, ctx);
        if url.trim().is_empty() {
            self.probe_response = ProbeResponseState::Failed("Enter a URL first.".to_string());
            ctx.notify();
            return;
        }
        let request = ProbeRequest {
            method: self.probe_method.clone(),
            url: url.trim().to_string(),
            headers: self.probe_field_text(&self.probe_headers_input, ctx),
            body: self.probe_field_text(&self.probe_body_input, ctx),
        };

        self.probe_request_generation = self.probe_request_generation.wrapping_add(1);
        let generation = self.probe_request_generation;
        let record = request.clone();
        self.probe_response = ProbeResponseState::InFlight;
        self.probe_show_response_headers = false;
        ctx.notify();

        ctx.spawn(
            async move { send_request(request).await },
            move |me, result, ctx| {
                // A second Send while the first was in flight wins; this guard
                // stops the slower response from landing on top of it.
                if me.probe_request_generation != generation {
                    return;
                }
                match result {
                    Ok(response) => {
                        me.push_probe_history(&record, Some(response.status), response.elapsed_ms);
                        me.probe_response = ProbeResponseState::Ready(Box::new(response));
                    }
                    Err(error) => {
                        me.push_probe_history(&record, None, 0);
                        me.probe_response = ProbeResponseState::Failed(format!("{error}"));
                    }
                }
                ctx.notify();
            },
        );
    }

    fn push_probe_history(
        &mut self,
        request: &ProbeRequest,
        status: Option<u16>,
        elapsed_ms: u128,
    ) {
        self.probe_history.insert(
            0,
            ProbeHistoryEntry {
                method: request.method.clone(),
                url: request.url.clone(),
                headers: request.headers.clone(),
                body: request.body.clone(),
                status,
                elapsed_ms,
            },
        );
        self.probe_history.truncate(PROBE_HISTORY_LIMIT);
    }

    pub fn replay_probe_history(&mut self, index: usize, ctx: &mut ViewContext<Self>) {
        let Some(entry) = self.probe_history.get(index).cloned() else {
            return;
        };
        self.probe_method = entry.method;
        self.probe_selected_route = None;
        self.set_probe_field(&self.probe_url, &entry.url, ctx);
        self.set_probe_field(&self.probe_headers_input, &entry.headers, ctx);
        self.set_probe_field(&self.probe_body_input, &entry.body, ctx);
        ctx.notify();
    }

    pub fn clear_probe_history(&mut self, ctx: &mut ViewContext<Self>) {
        self.probe_history.clear();
        ctx.notify();
    }

    pub fn copy_probe_response(&mut self, ctx: &mut ViewContext<Self>) {
        let body = match &self.probe_response {
            ProbeResponseState::Ready(response) => response.body.clone(),
            ProbeResponseState::Failed(error) => error.clone(),
            _ => return,
        };
        ctx.clipboard().write(ClipboardContent::plain_text(body));
    }

    /// Open Probe already aimed at an endpoint the user clicked on the Canvas.
    ///
    /// The seed comes from the Canvas graph rather than the survey, so the
    /// request is filled in immediately instead of after the port scan — the
    /// button would otherwise feel like it did nothing.
    pub fn probe_canvas_node(
        &mut self,
        node_id: &str,
        project_root: Option<PathBuf>,
        ctx: &mut ViewContext<Self>,
    ) {
        let seed = match &self.project_canvas_state {
            ProjectCanvasState::Ready(graph) => graph.node(node_id).and_then(|node| {
                let route = node.route.clone()?;
                Some((
                    node.method.clone().unwrap_or_else(|| "GET".to_string()),
                    route,
                    node.request_body.clone(),
                ))
            }),
            _ => None,
        };

        self.open_api_probe(project_root, ctx);

        if let Some((method, route, body_hint)) = seed {
            // Route ids are shared with the Canvas, so the sidebar highlights
            // this endpoint as soon as the survey lands behind us.
            self.probe_selected_route = Some(node_id.to_string());
            match &self.probe_state {
                ProbeState::Ready(survey) => {
                    let base = survey.base_url();
                    self.probe_method = method.clone();
                    self.seed_probe_url(&build_url(&base, &route), ctx);
                    if method_takes_body(&method) {
                        self.seed_probe_body(body_hint.as_deref(), ctx);
                    }
                }
                // The survey is still running, so the port is not known. This
                // used to fall back to a hardcoded :3000 and never correct
                // itself once the real port arrived — which is exactly what
                // made a backend on :3210 get called on :3000. Hold the route
                // until `apply_probe_survey` can build it properly.
                _ => self.probe_pending_route = Some((method, route, body_hint)),
            }
        }
        ctx.notify();
    }

    // ──────────────────────── Code reading lenses ────────────────────────

    /// Recompute both lenses for the focused file.
    ///
    /// Architecture needs a project walk to answer "who imports this", which is
    /// the half you cannot see from inside the file — so the whole thing runs
    /// off the UI thread.
    pub fn open_lenses(&mut self, target: Option<(PathBuf, PathBuf)>, ctx: &mut ViewContext<Self>) {
        self.active_side_tool = GenesiSideTool::Lenses;
        self.lenses_generation = self.lenses_generation.wrapping_add(1);
        let generation = self.lenses_generation;
        let Some((root, file)) = target else {
            self.lenses = None;
            ctx.notify();
            return;
        };
        ctx.notify();

        ctx.spawn(
            async move {
                tokio::task::spawn_blocking(move || {
                    (architecture(&root, &file), performance(&root, &file))
                })
                .await
            },
            move |me, result, ctx| {
                if me.lenses_generation != generation {
                    return;
                }
                me.lenses = result.ok();
                ctx.notify();
            },
        );
    }

    pub fn show_performance_lens(&mut self, performance: bool, ctx: &mut ViewContext<Self>) {
        self.lens_performance = performance;
        ctx.notify();
    }

    fn render_lens_section(
        &self,
        appearance: &Appearance,
        label: String,
        body: String,
    ) -> Box<dyn Element> {
        Container::new(self.render_canvas_inspector_field(appearance, &label, body, false))
            .with_horizontal_padding(10.)
            .with_margin_bottom(10.)
            .finish()
    }

    fn render_lenses(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let mut content = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);

        content.add_child(
            Container::new(
                Flex::row()
                    .with_child(self.workspace_chip(
                        appearance,
                        "Architecture".to_string(),
                        None,
                        WorkspaceAction::ShowGenesiPerformanceLens(false),
                        !self.lens_performance,
                    ))
                    .with_child(self.workspace_chip(
                        appearance,
                        "Performance".to_string(),
                        None,
                        WorkspaceAction::ShowGenesiPerformanceLens(true),
                        self.lens_performance,
                    ))
                    .finish(),
            )
            .with_uniform_padding(10.)
            .finish(),
        );

        let Some((architecture, performance)) = &self.lenses else {
            content.add_child(
                Container::new(self.label_text(
                    appearance,
                    "Open a source file to read it through a lens.".to_string(),
                    12.,
                    muted,
                    true,
                ))
                .with_uniform_padding(10.)
                .finish(),
            );
            return content.finish();
        };

        if self.lens_performance {
            content.add_child(
                Container::new(self.label_text(
                    appearance,
                    performance.summary(),
                    12.,
                    theme.main_text_color(theme.background()).into(),
                    true,
                ))
                .with_horizontal_padding(10.)
                .with_margin_bottom(8.)
                .finish(),
            );
            for finding in &performance.findings {
                let accent: ColorU = match finding.severity {
                    Severity::Scaling => theme.terminal_colors().normal.red.into(),
                    Severity::Waste => theme.terminal_colors().normal.yellow.into(),
                    Severity::Note => muted,
                };
                content.add_child(
                    Container::new(
                        Flex::column()
                            .with_cross_axis_alignment(CrossAxisAlignment::Start)
                            .with_child(self.label_text(
                                appearance,
                                format!("line {} · {}", finding.line, finding.title),
                                12.,
                                accent,
                                true,
                            ))
                            .with_child(self.mono_text(
                                appearance,
                                finding.snippet.clone(),
                                11.,
                                theme.main_text_color(theme.background()).into(),
                                true,
                            ))
                            .with_child(self.label_text(
                                appearance,
                                finding.detail.clone(),
                                11.,
                                muted,
                                true,
                            ))
                            .finish(),
                    )
                    .with_uniform_padding(10.)
                    .with_margin_left(10.)
                    .with_margin_right(10.)
                    .with_margin_bottom(6.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                    .with_background(theme.surface_1())
                    .with_border(Border::all(1.).with_border_fill(theme.outline()))
                    .finish(),
                );
            }
            return content.finish();
        }

        content.add_child(self.render_lens_section(
            appearance,
            "Position".to_string(),
            format!("{}\n{}", architecture.layer.label(), architecture.summary()),
        ));
        content.add_child(self.render_lens_section(
            appearance,
            format!("Depended on by ({})", architecture.fan_in()),
            if architecture.dependents.is_empty() {
                "Nothing in the project imports this file".to_string()
            } else {
                architecture
                    .dependents
                    .iter()
                    .map(|path| path.display().to_string())
                    .collect::<Vec<_>>()
                    .join("\n")
            },
        ));
        content.add_child(self.render_lens_section(
            appearance,
            format!(
                "Project imports ({})",
                architecture.internal_dependencies.len()
            ),
            if architecture.internal_dependencies.is_empty() {
                "None".to_string()
            } else {
                architecture.internal_dependencies.join("\n")
            },
        ));
        content.add_child(self.render_lens_section(
            appearance,
            format!(
                "External packages ({})",
                architecture.external_dependencies.len()
            ),
            if architecture.external_dependencies.is_empty() {
                "None".to_string()
            } else {
                architecture.external_dependencies.join(", ")
            },
        ));
        content.add_child(self.render_lens_section(
            appearance,
            format!("Exports ({})", architecture.exports.len()),
            if architecture.exports.is_empty() {
                "Nothing exported".to_string()
            } else {
                architecture.exports.join(", ")
            },
        ));
        content.finish()
    }

    pub fn open_project_canvas(
        &mut self,
        project_root: Option<PathBuf>,
        ctx: &mut ViewContext<Self>,
    ) {
        self.active_side_tool = GenesiSideTool::Canvas;
        self.refresh_project_canvas(project_root, ctx);
    }

    pub fn refresh_project_canvas(
        &mut self,
        project_root: Option<PathBuf>,
        ctx: &mut ViewContext<Self>,
    ) {
        self.project_canvas_generation = self.project_canvas_generation.wrapping_add(1);
        let generation = self.project_canvas_generation;
        let Some(root) = project_root else {
            self.project_canvas_state = ProjectCanvasState::NoProject;
            self.selected_canvas_node = None;
            Arc::make_mut(&mut self.project_canvas_positions).clear();
            self.project_canvas_drag = None;
            self.reset_project_canvas_drag_sampling();
            ctx.notify();
            return;
        };

        self.agent_root = Some(root.clone());
        self.project_canvas_state = ProjectCanvasState::Loading(root.clone());
        self.selected_canvas_node = None;
        ctx.notify();

        ctx.spawn(
            async move { tokio::task::spawn_blocking(move || analyze_project(&root)).await },
            move |me, result, ctx| {
                if me.project_canvas_generation != generation {
                    return;
                }
                match result {
                    Ok(graph) => {
                        me.selected_canvas_node = graph
                            .nodes
                            .iter()
                            .find(|node| node.kind == CanvasNodeKind::Page)
                            .or_else(|| {
                                graph
                                    .nodes
                                    .iter()
                                    .find(|node| node.kind == CanvasNodeKind::Endpoint)
                            })
                            .or_else(|| graph.nodes.first())
                            .map(|node| node.id.clone());
                        me.project_canvas_positions =
                            Arc::new(Self::auto_arranged_project_canvas_positions(&graph));
                        me.project_canvas_pan = vec2f(42., 42.);
                        me.project_canvas_zoom = if graph.nodes.len() > 14 { 0.75 } else { 1. };
                        me.project_canvas_drag = None;
                        me.reset_project_canvas_drag_sampling();
                        me.project_canvas_state = ProjectCanvasState::Ready(Arc::new(graph));
                    }
                    Err(error) => {
                        me.project_canvas_state = ProjectCanvasState::Error(format!(
                            "The background analyzer stopped unexpectedly: {error}"
                        ));
                        me.selected_canvas_node = None;
                        Arc::make_mut(&mut me.project_canvas_positions).clear();
                        me.project_canvas_drag = None;
                        me.reset_project_canvas_drag_sampling();
                    }
                }
                ctx.notify();
            },
        );
    }

    pub fn select_project_canvas_node(&mut self, id: &str, ctx: &mut ViewContext<Self>) {
        let exists = match &self.project_canvas_state {
            ProjectCanvasState::Ready(graph) => graph.node(id).is_some(),
            _ => false,
        };
        if exists {
            self.selected_canvas_node = Some(id.to_string());
            ctx.notify();
        }
    }

    pub fn keep_pending_edits(&mut self, ctx: &mut ViewContext<Self>) {
        let paths = self
            .pending_edits
            .iter()
            .map(|edit| edit.path.clone())
            .collect::<Vec<_>>();
        for path in paths {
            self.update_open_editor_diff(&path, None, ctx);
        }
        self.pending_edits.clear();
        self.review_expanded = false;
        self.selected_review_path = None;
        self.persist_mempalace(ctx, false);
        ctx.notify();
    }

    pub fn undo_pending_edits(&mut self, ctx: &mut ViewContext<Self>) {
        let edits = std::mem::take(&mut self.pending_edits);
        if let Some(root) = self.agent_root.clone() {
            for edit in edits {
                let Some(path) = local_agent::resolve_in_project(&root, &edit.path) else {
                    continue;
                };
                match edit.original {
                    Some(content) => {
                        let _ = std::fs::write(&path, content);
                    }
                    None => {
                        let _ = std::fs::remove_file(&path);
                    }
                }
            }
        }
        self.review_expanded = false;
        self.selected_review_path = None;
        self.persist_mempalace(ctx, false);
        ctx.notify();
    }

    pub fn render_review_sidebar(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let side_tool_close_color: ColorU = theme.sub_text_color(theme.background()).into();
        let mut root = Flex::column()
            .with_main_axis_size(MainAxisSize::Max)
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch);

        let header = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_child(
                Expanded::new(
                    1.,
                    self.label_text(
                        appearance,
                        match self.active_side_tool {
                            GenesiSideTool::Review => "Review",
                            GenesiSideTool::Canvas => "Canvas",
                            GenesiSideTool::Lenses => "Lenses",
                        }
                        .to_string(),
                        TITLE_FONT_SIZE,
                        theme.main_text_color(theme.background()).into(),
                        false,
                    ),
                )
                .finish(),
            )
            .with_child(self.workspace_chip(
                appearance,
                "Review".to_string(),
                None,
                WorkspaceAction::OpenGenesiReviewTool,
                self.active_side_tool == GenesiSideTool::Review,
            ))
            .with_child(self.workspace_chip(
                appearance,
                "Canvas".to_string(),
                None,
                WorkspaceAction::OpenGenesiCanvasTool,
                self.active_side_tool == GenesiSideTool::Canvas,
            ))
            .with_child(self.workspace_chip(
                appearance,
                "Lenses".to_string(),
                None,
                WorkspaceAction::OpenGenesiLensesTool,
                self.active_side_tool == GenesiSideTool::Lenses,
            ))
            // The panel had no way out from inside itself: once open, the only
            // way to dismiss it was the toolbar button that opened it.
            .with_child(
                EventHandler::new(
                    Container::new(
                        ConstrainedBox::new(
                            Icon::new("bundled/svg/x-close.svg", side_tool_close_color).finish(),
                        )
                        .with_width(12.)
                        .with_height(12.)
                        .finish(),
                    )
                    .with_uniform_padding(6.)
                    .with_margin_left(4.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
                    .finish(),
                )
                .on_left_mouse_down(move |ctx, _, _| {
                    ctx.dispatch_typed_action(WorkspaceAction::ToggleGenesiToolsPanel);
                    DispatchEventResult::StopPropagation
                })
                .finish(),
            )
            .finish();
        root.add_child(
            Container::new(header)
                .with_uniform_padding(10.)
                .with_border(Border::bottom(1.).with_border_fill(theme.outline()))
                .finish(),
        );

        if self.active_side_tool == GenesiSideTool::Lenses {
            root.add_child(
                Expanded::new(
                    1.,
                    ClippedScrollable::vertical(
                        self.lenses_scroll.clone(),
                        self.render_lenses(appearance),
                        ScrollbarWidth::Auto,
                        theme.disabled_ui_text_color().into(),
                        theme.active_ui_text_color().into(),
                        Fill::None,
                    )
                    .finish(),
                )
                .finish(),
            );
            return root.finish();
        }

        if self.active_side_tool == GenesiSideTool::Canvas {
            root.add_child(
                Expanded::new(
                    1.,
                    ClippedScrollable::vertical(
                        self.review_sidebar_scroll.clone(),
                        self.render_project_canvas(appearance),
                        ScrollbarWidth::Auto,
                        theme.disabled_ui_text_color().into(),
                        theme.active_ui_text_color().into(),
                        Fill::None,
                    )
                    .finish(),
                )
                .finish(),
            );
            return root.finish();
        }

        let selected_preview = self.selected_review_preview();
        if self.pending_edits.is_empty() && selected_preview.is_none() {
            root.add_child(
                Expanded::new(
                    1.,
                    Container::new(self.label_text(
                        appearance,
                        "No changes to review yet.".to_string(),
                        BODY_FONT_SIZE,
                        theme.disabled_text_color(theme.background()).into(),
                        true,
                    ))
                    .with_uniform_padding(16.)
                    .finish(),
                )
                .finish(),
            );
            return root.finish();
        }

        let selected_path = selected_preview
            .as_ref()
            .map(|preview| preview.0.to_string())
            .unwrap_or_default();
        let mut files = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);
        let file_rows: Vec<(&str, u32, u32)> = if self.pending_edits.is_empty() {
            selected_preview
                .as_ref()
                .map(|preview| vec![(preview.0, preview.1, preview.2)])
                .unwrap_or_default()
        } else {
            self.pending_edits
                .iter()
                .map(|edit| (edit.path.as_str(), edit.added, edit.removed))
                .collect()
        };
        for (edit_path, added, removed) in file_rows {
            let path = edit_path.to_string();
            let row = Flex::row()
                .with_cross_axis_alignment(CrossAxisAlignment::Center)
                .with_child(
                    Expanded::new(
                        1.,
                        self.label_text(
                            appearance,
                            edit_path.to_string(),
                            CHIP_FONT_SIZE,
                            theme.main_text_color(theme.background()).into(),
                            false,
                        ),
                    )
                    .finish(),
                )
                .with_child(self.label_text(
                    appearance,
                    format!("+{added}"),
                    CHIP_FONT_SIZE,
                    genesi_green(),
                    false,
                ))
                .with_child(
                    Container::new(self.label_text(
                        appearance,
                        format!("-{removed}"),
                        CHIP_FONT_SIZE,
                        theme.ui_error_color().into(),
                        false,
                    ))
                    .with_margin_left(5.)
                    .finish(),
                )
                .finish();
            files.add_child(
                EventHandler::new(
                    Container::new(row)
                        .with_horizontal_padding(10.)
                        .with_vertical_padding(8.)
                        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
                        .with_background_color(if edit_path == selected_path {
                            ColorU::new(255, 255, 255, 18)
                        } else {
                            ColorU::new(0, 0, 0, 0)
                        })
                        .finish(),
                )
                .on_left_mouse_down(move |ctx, _, _| {
                    ctx.dispatch_typed_action(WorkspaceAction::SelectGenesiReviewFile(
                        path.clone(),
                    ));
                    DispatchEventResult::StopPropagation
                })
                .finish(),
            );
        }
        root.add_child(
            Container::new(files.finish())
                .with_uniform_padding(8.)
                .with_border(Border::bottom(1.).with_border_fill(theme.outline()))
                .finish(),
        );

        if let Some((_, _, _, diff_preview)) = selected_preview {
            root.add_child(
                Expanded::new(
                    1.,
                    ClippedScrollable::vertical(
                        self.review_sidebar_scroll.clone(),
                        Container::new(self.render_diff_preview(appearance, diff_preview))
                            .with_uniform_padding(8.)
                            .finish(),
                        ScrollbarWidth::Auto,
                        theme.disabled_ui_text_color().into(),
                        theme.active_ui_text_color().into(),
                        Fill::None,
                    )
                    .finish(),
                )
                .finish(),
            );
        }

        if !self.pending_edits.is_empty() {
            root.add_child(
                Container::new(
                    Flex::row()
                        .with_main_axis_alignment(MainAxisAlignment::End)
                        .with_child(self.workspace_chip(
                            appearance,
                            "Undo".to_string(),
                            None,
                            WorkspaceAction::UndoGenesiEdits,
                            false,
                        ))
                        .with_child(self.workspace_chip(
                            appearance,
                            "Keep All".to_string(),
                            None,
                            WorkspaceAction::KeepGenesiEdits,
                            true,
                        ))
                        .finish(),
                )
                .with_uniform_padding(10.)
                .with_border(Border::top(1.).with_border_fill(theme.outline()))
                .finish(),
            );
        }

        root.finish()
    }

    fn reset_current_chat(&mut self) {
        self.messages.clear();
        self.error = None;
        self.pending_tool = None;
        self.pending_edits.clear();
        self.review_expanded = false;
        self.selected_review_path = None;
        self.agent_messages.clear();
        self.agent_root = None;
        self.agent_model.clear();
        self.agent_step = 0;
        self.agent_step_buffer.clear();
        self.agent_tool_summary.clear();
        self.in_flight = false;
        self.steering.clear();
        self.current_turn += 1;
    }

    fn selected_review_edit(&self) -> Option<&PendingEdit> {
        self.selected_review_path
            .as_ref()
            .and_then(|path| self.pending_edits.iter().find(|edit| &edit.path == path))
            .or_else(|| self.pending_edits.first())
    }

    /// Returns the selected diff whether it is still pending or is an immutable
    /// historical snapshot stored on the tool card in the conversation.
    fn selected_review_preview(&self) -> Option<(&str, u32, u32, &[DiffPreviewLine])> {
        if let Some(edit) = self.selected_review_edit() {
            return Some((
                edit.path.as_str(),
                edit.added,
                edit.removed,
                edit.diff_preview.as_slice(),
            ));
        }

        let selected_path = self.selected_review_path.as_deref()?;
        self.messages.iter().rev().find_map(|entry| {
            let path = entry.tool_title.as_deref()?;
            if path != selected_path {
                return None;
            }
            let preview = entry.diff_preview.as_deref()?;
            let (added, removed) = entry.diff_stat.unwrap_or_default();
            Some((path, added, removed, preview))
        })
    }

    fn collapse_other_diffs(&mut self, keep_index: usize) {
        for (index, entry) in self.messages.iter_mut().enumerate() {
            if index != keep_index
                && entry.role == ChatRole::Tool
                && entry
                    .diff_preview
                    .as_ref()
                    .is_some_and(|lines| !lines.is_empty())
            {
                entry.collapsed = true;
            }
        }
    }

    fn render_diff_preview(
        &self,
        appearance: &Appearance,
        lines: &[DiffPreviewLine],
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let mut column = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);

        for line in lines {
            let (prefix, bg, fg) = match line.kind {
                DiffPreviewLineKind::Context => (
                    " ",
                    ColorU::new(0, 0, 0, 0),
                    theme.disabled_text_color(theme.background()).into(),
                ),
                DiffPreviewLineKind::Added => ("+", ColorU::new(15, 143, 106, 36), genesi_green()),
                DiffPreviewLineKind::Removed => (
                    "-",
                    ColorU::new(181, 68, 68, 36),
                    theme.ui_error_color().into(),
                ),
            };

            let old_line = line
                .old_line
                .map(|value| value.to_string())
                .unwrap_or_default();
            let new_line = line
                .new_line
                .map(|value| value.to_string())
                .unwrap_or_default();

            let row = Flex::row()
                .with_cross_axis_alignment(CrossAxisAlignment::Center)
                .with_child(
                    ConstrainedBox::new(
                        Container::new(self.mono_text(
                            appearance,
                            format!("{old_line:>4}"),
                            MONO_FONT_SIZE,
                            theme.disabled_text_color(theme.background()).into(),
                            false,
                        ))
                        .finish(),
                    )
                    .with_width(34.)
                    .finish(),
                )
                .with_child(
                    ConstrainedBox::new(
                        Container::new(self.mono_text(
                            appearance,
                            format!("{new_line:>4}"),
                            MONO_FONT_SIZE,
                            theme.disabled_text_color(theme.background()).into(),
                            false,
                        ))
                        .finish(),
                    )
                    .with_width(34.)
                    .finish(),
                )
                .with_child(
                    ConstrainedBox::new(
                        Container::new(self.mono_text(
                            appearance,
                            prefix.to_string(),
                            MONO_FONT_SIZE,
                            fg,
                            false,
                        ))
                        .finish(),
                    )
                    .with_width(12.)
                    .finish(),
                )
                .with_child(
                    Expanded::new(
                        1.,
                        self.mono_text(appearance, line.text.clone(), MONO_FONT_SIZE, fg, true),
                    )
                    .finish(),
                )
                .finish();

            column.add_child(
                Container::new(row)
                    .with_horizontal_padding(8.)
                    .with_vertical_padding(2.)
                    .with_background_color(bg)
                    .finish(),
            );
        }

        Container::new(column.finish())
            .with_uniform_padding(6.)
            .with_margin_top(4.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
            .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
            .with_background_color(genesi_card_surface())
            .finish()
    }

    /// The panel header keeps only the assistant title and utility controls.
    /// The global Vibe/IDE switch belongs to the app title bar.
    fn render_header(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();

        let mut row = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_main_axis_size(MainAxisSize::Max)
            .with_child(
                Container::new(
                    ConstrainedBox::new(
                        Icon::new("bundled/svg/sparkle.svg", genesi_green()).finish(),
                    )
                    .with_width(13.)
                    .with_height(13.)
                    .finish(),
                )
                .with_margin_right(6.)
                .finish(),
            )
            .with_child(self.label_text(
                appearance,
                "Genesi Code",
                TITLE_FONT_SIZE,
                theme.active_ui_text_color().into(),
                false,
            ))
            // The CURRENT chat's title. This used to be the hardcoded words "New
            // chat", which never changed and was not a button — clicking it did
            // nothing because there was nothing there to click.
            .with_child(
                Container::new(self.label_text(
                    appearance,
                    Self::chat_title_from_entries(&self.messages),
                    CHIP_FONT_SIZE,
                    theme.disabled_text_color(theme.background()).into(),
                    false,
                ))
                .with_margin_left(7.)
                .finish(),
            )
            .with_child(Shrinkable::new(1., Empty::new().finish()).finish())
            // Starting a fresh chat had a working action and no way to reach it.
            .with_child(self.utility_icon_button(
                "bundled/svg/clock-rewind.svg",
                LocalAiChatAction::ToggleChatPicker,
                true,
            ))
            .with_child(self.utility_icon_button(
                "bundled/svg/add.svg",
                LocalAiChatAction::NewChat,
                !self.messages.is_empty(),
            ))
            .with_child(self.utility_icon_button(
                "bundled/svg/refresh-cw-04.svg",
                LocalAiChatAction::Refresh,
                true,
            ));
        row.add_child(self.utility_icon_button(
            "bundled/svg/trash-02.svg",
            LocalAiChatAction::Clear,
            !self.messages.is_empty(),
        ));
        row.finish()
    }

    fn utility_icon_button(
        &self,
        icon_path: &'static str,
        action: LocalAiChatAction,
        enabled: bool,
    ) -> Box<dyn Element> {
        let icon_color = if enabled {
            ColorU::new(222, 225, 231, 255)
        } else {
            ColorU::new(126, 130, 138, 255)
        };
        let button = Container::new(
            ConstrainedBox::new(Icon::new(icon_path, icon_color).finish())
                .with_width(13.)
                .with_height(13.)
                .finish(),
        )
        .with_uniform_padding(7.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(7.)))
        .with_margin_left(4.)
        .finish();

        if !enabled {
            return button;
        }
        EventHandler::new(button)
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    /// The bottom control strip, sitting just above the compose box like a real
    /// IDE: a click-to-open AI selector, the Turbo toggle next to it, and the
    /// agent toggles. The model picker itself is a popup (see
    /// [`Self::render_model_picker`]) so the user clicks once and chooses.
    fn render_control_strip(&self, appearance: &Appearance) -> Box<dyn Element> {
        let model_name = self
            .current_model()
            .map(|model| self.model_label(&model))
            .unwrap_or_else(|| "no model".to_string());
        let model_name = truncate_middle(&model_name, MODEL_LABEL_MAX_CHARS);
        let selector_label = if self.cloud_active {
            format!("{}: {model_name}", self.cloud.provider.label())
        } else if self.endpoint == LocalEndpoint::Turbo {
            format!("Turbo: {model_name}")
        } else {
            format!("AI: {model_name}")
        };

        // The compose bar carries exactly three controls, matching the design:
        // an attach affordance on the left, then the mode and model selectors on
        // the right. Everything that used to sit here as its own chip (Agent,
        // AUTO, Turbo, AI Mode) is now either folded into the mode selector or
        // lives in the model picker, so the row stops looking like a toolbar.
        Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_main_axis_size(MainAxisSize::Max)
            // The paperclip does what a paperclip does everywhere else: opens a
            // file picker. It used to toggle whether the editor's current file
            // was auto-attached, which is a different idea entirely — that toggle
            // now lives on its own chip in the row above.
            .with_child(self.icon_toggle_button(
                appearance,
                "bundled/svg/paperclip.svg",
                LocalAiChatAction::PickAttachments,
                !self.attachments.is_empty(),
            ))
            .with_child(Shrinkable::new(1., Empty::new().finish()).finish())
            .with_child(self.selector_button(
                appearance,
                self.chat_mode().label().to_string(),
                LocalAiChatAction::ToggleModePicker,
                self.mode_picker_open,
            ))
            .with_child(self.selector_button(
                appearance,
                selector_label,
                LocalAiChatAction::ToggleModelPicker,
                self.model_picker_open,
            ))
            // Send lives at the end of this row, i.e. the bottom-right corner of
            // the compose box.
            .with_child(
                Container::new(self.render_prompt_action_button(appearance))
                    .with_margin_left(6.)
                    .finish(),
            )
            .finish()
    }

    /// The current mode, derived from the two flags it drives. Keeping the flags
    /// as the source of truth means every existing behaviour check
    /// (`agent_mode`, `auto_approve`) keeps working untouched.
    fn chat_mode(&self) -> ChatMode {
        match (self.agent_mode, self.auto_approve) {
            (false, _) => ChatMode::Chat,
            (true, false) => ChatMode::Build,
            (true, true) => ChatMode::Auto,
        }
    }

    fn set_chat_mode(&mut self, mode: ChatMode) {
        match mode {
            ChatMode::Chat => self.agent_mode = false,
            ChatMode::Build => {
                self.agent_mode = true;
                self.auto_approve = false;
            }
            ChatMode::Auto => {
                self.agent_mode = true;
                self.auto_approve = true;
            }
        }
    }

    /// The row of reference chips above the input: the editor's own file (when
    /// auto-attach is on) followed by everything the user pinned by hand. Absent
    /// entirely when there is nothing attached and auto-attach is off, so a plain
    /// chat keeps a clean compose box.
    fn render_attachment_row(&self, appearance: &Appearance) -> Option<Box<dyn Element>> {
        // A column, not a row: the panel is narrow and chip names are long, so
        // one reference per line always fits where a row would clip.
        let mut row = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Start);
        // The implicit editor context, made visible: it behaves like every other
        // chip, so dropping it is the same gesture as dropping a file — and when
        // it is off, the same chip is how you get it back.
        row.add_child(self.render_attachment_chip(
            appearance,
            "bundled/svg/file-code-02.svg",
            if self.attach_context {
                "Editor file".to_string()
            } else {
                "Editor file off".to_string()
            },
            LocalAiChatAction::ToggleAttachContext,
            !self.attach_context,
        ));
        for (index, attachment) in self.attachments.iter().enumerate() {
            let icon = match attachment.kind {
                AttachmentKind::Image => "bundled/svg/image-01.svg",
                AttachmentKind::Text => "bundled/svg/file-06.svg",
            };
            row.add_child(self.render_attachment_chip(
                appearance,
                icon,
                truncate_middle(&attachment.name, 28),
                LocalAiChatAction::RemoveAttachment(index),
                false,
            ));
        }
        Some(
            Container::new(row.finish())
                .with_horizontal_padding(4.)
                .with_padding_bottom(6.)
                .finish(),
        )
    }

    /// One reference chip: icon, name, and a trailing × that removes it. An `off`
    /// chip is the inverse — greyed, with a ＋ that puts the reference back.
    fn render_attachment_chip(
        &self,
        appearance: &Appearance,
        icon: &'static str,
        label: String,
        remove: LocalAiChatAction,
        off: bool,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let text_color: ColorU = if off {
            theme.disabled_text_color(theme.background()).into()
        } else {
            theme.active_ui_text_color().into()
        };
        let icon_color: ColorU = theme.sub_text_color(theme.background()).into();
        let trailing = if off {
            "bundled/svg/add.svg"
        } else {
            "bundled/svg/x-close.svg"
        };

        let chip = Container::new(
            Flex::row()
                .with_cross_axis_alignment(CrossAxisAlignment::Center)
                .with_child(
                    Container::new(
                        ConstrainedBox::new(Icon::new(icon, icon_color).finish())
                            .with_width(12.)
                            .with_height(12.)
                            .finish(),
                    )
                    .with_margin_right(5.)
                    .finish(),
                )
                .with_child(self.label_text(appearance, label, CHIP_FONT_SIZE, text_color, false))
                .with_child(
                    Container::new(
                        ConstrainedBox::new(Icon::new(trailing, icon_color).finish())
                            .with_width(10.)
                            .with_height(10.)
                            .finish(),
                    )
                    .with_margin_left(6.)
                    .finish(),
                )
                .finish(),
        )
        .with_horizontal_padding(7.)
        .with_vertical_padding(4.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(7.)))
        .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
        .with_background_color(genesi_card_surface())
        .with_margin_right(5.)
        .with_margin_top(4.);

        // The whole chip is the remove target: it is small, and a 10px × alone
        // would be a miss-prone hit box.
        EventHandler::new(chip.finish())
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(remove.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    /// A square, icon-only toggle — the attach affordance in the design is an
    /// icon with no label, tinted when active.
    fn icon_toggle_button(
        &self,
        appearance: &Appearance,
        icon: &'static str,
        action: LocalAiChatAction,
        active: bool,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let color: ColorU = if active {
            genesi_green()
        } else {
            theme.sub_text_color(theme.background()).into()
        };
        let mut container = Container::new(
            ConstrainedBox::new(Icon::new(icon, color).finish())
                .with_width(15.)
                .with_height(15.)
                .finish(),
        )
        .with_uniform_padding(6.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
        .with_margin_top(2.);
        if active {
            container = container.with_background_color(green_tint());
        }

        EventHandler::new(container.finish())
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    /// The Chat / Build / Auto popup, opened from the mode selector.
    fn render_mode_picker(&self, appearance: &Appearance) -> Option<Box<dyn Element>> {
        if !self.mode_picker_open {
            return None;
        }
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let current = self.chat_mode();

        let mut list = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);
        for mode in [ChatMode::Chat, ChatMode::Build, ChatMode::Auto] {
            let selected = mode == current;
            list.add_child(self.picker_row(
                appearance,
                format!("{}{}", if selected { "●  " } else { "○  " }, mode.label()),
                LocalAiChatAction::SetChatMode(mode),
                selected,
            ));
            list.add_child(
                Container::new(self.label_text(
                    appearance,
                    mode.description(),
                    CHIP_FONT_SIZE,
                    muted,
                    true,
                ))
                .with_horizontal_padding(22.)
                .with_padding_bottom(4.)
                .finish(),
            );
        }
        Some(self.popup_surface(list.finish()))
    }

    /// The saved-conversation list.
    ///
    /// It existed only in Vibe mode, so in the IDE the panel had no history at
    /// all — a chat was reachable only while it was the active one. The data and
    /// the open/delete calls were already there; only this list was missing.
    fn render_chat_picker(&self, appearance: &Appearance) -> Option<Box<dyn Element>> {
        if !self.chat_picker_open {
            return None;
        }
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let summaries = self.chat_summaries();

        let mut list = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);
        if summaries.is_empty() {
            list.add_child(
                Container::new(self.label_text(
                    appearance,
                    "No saved chats yet.".to_string(),
                    CHIP_FONT_SIZE,
                    muted,
                    true,
                ))
                .with_uniform_padding(10.)
                .finish(),
            );
            return Some(self.popup_surface(list.finish()));
        }

        for summary in summaries {
            list.add_child(
                Flex::row()
                    .with_cross_axis_alignment(CrossAxisAlignment::Center)
                    .with_child(
                        Expanded::new(
                            1.,
                            self.picker_row(
                                appearance,
                                format!(
                                    "{}{}",
                                    if summary.is_active { "●  " } else { "○  " },
                                    summary.title
                                ),
                                LocalAiChatAction::PickChat(summary.id.clone()),
                                summary.is_active,
                            ),
                        )
                        .finish(),
                    )
                    .with_child(self.utility_icon_button(
                        "bundled/svg/trash-02.svg",
                        LocalAiChatAction::DeleteChat(summary.id.clone()),
                        true,
                    ))
                    .finish(),
            );
        }
        Some(self.popup_surface(list.finish()))
    }

    /// Shared framing for the popups that float above the compose box.
    fn popup_surface(&self, contents: Box<dyn Element>) -> Box<dyn Element> {
        Container::new(contents)
            .with_background_color(genesi_card_surface())
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(12.)))
            .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
            .with_uniform_padding(6.)
            .with_margin_bottom(6.)
            .finish()
    }

    /// A quiet dropdown-style label used as the AI selector. It intentionally
    /// avoids the old bordered pill style so long model names do not dominate.
    fn selector_button(
        &self,
        appearance: &Appearance,
        label: String,
        action: LocalAiChatAction,
        open: bool,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let text_color: ColorU = theme.active_ui_text_color().into();
        let chevron = if open {
            "bundled/svg/chevron-up.svg"
        } else {
            "bundled/svg/chevron-down.svg"
        };
        let container = Container::new(
            Flex::row()
                .with_cross_axis_alignment(CrossAxisAlignment::Center)
                .with_child(self.label_text(appearance, label, BODY_FONT_SIZE, text_color, false))
                .with_child(
                    Container::new(
                        ConstrainedBox::new(Icon::new(chevron, text_color).finish())
                            .with_width(12.)
                            .with_height(12.)
                            .finish(),
                    )
                    .with_margin_left(6.)
                    .finish(),
                )
                .finish(),
        )
        // Text only: no frame and no fill, as in the reference. A tint appears
        // solely while the popup is open, so the trigger still reads as active.
        .with_horizontal_padding(8.)
        .with_vertical_padding(5.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
        .with_background_color(if open {
            green_tint()
        } else {
            ColorU::transparent_black()
        })
        .with_margin_right(4.)
        .with_margin_top(2.);

        EventHandler::new(container.finish())
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    fn render_vibe_lab_strip(&self, appearance: &Appearance) -> Box<dyn Element> {
        const SOUNDSCAPES: [&str; 4] = ["Rain", "Cafe", "Forest", "Lo-fi"];
        const SWITCHES: [&str; 3] = ["Red", "Brown", "Blue"];

        let sound = SOUNDSCAPES[self.soundscape_index % SOUNDSCAPES.len()];
        let switch = SWITCHES[self.keyboard_switch_index % SWITCHES.len()];

        Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_child(self.chip(
                appearance,
                if self.soundscape_enabled {
                    format!("Sound {sound}")
                } else {
                    "Sound off".to_string()
                },
                LocalAiChatAction::ToggleSoundscape,
                self.soundscape_enabled,
            ))
            .with_child(self.chip(
                appearance,
                "Cycle sound".to_string(),
                LocalAiChatAction::CycleSoundscape,
                false,
            ))
            .with_child(self.chip(
                appearance,
                if self.keyboard_asmr_enabled {
                    format!("Keys {switch}")
                } else {
                    "Keys off".to_string()
                },
                LocalAiChatAction::ToggleKeyboardAsmr,
                self.keyboard_asmr_enabled,
            ))
            .with_child(self.chip(
                appearance,
                "Cycle keys".to_string(),
                LocalAiChatAction::CycleKeyboardSwitch,
                false,
            ))
            .finish()
    }

    fn render_prompt_action_button(&self, _appearance: &Appearance) -> Box<dyn Element> {
        let (icon_path, action, fill, icon_color) = if self.in_flight {
            (
                "bundled/svg/stop-filled.svg",
                LocalAiChatAction::Stop,
                ColorU::new(117, 34, 34, 255),
                ColorU::new(255, 214, 214, 255),
            )
        } else {
            (
                "bundled/svg/send.svg",
                LocalAiChatAction::SubmitPromptInput,
                genesi_green(),
                ColorU::white(),
            )
        };

        let button = Container::new(
            ConstrainedBox::new(Icon::new(icon_path, icon_color).finish())
                .with_width(14.)
                .with_height(14.)
                .finish(),
        )
        .with_horizontal_padding(8.)
        .with_vertical_padding(8.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(10.)))
        .with_border(Border::all(1.).with_border_color(if self.in_flight {
            ColorU::new(181, 68, 68, 90)
        } else {
            green_soft()
        }))
        .with_background_color(fill)
        .finish();

        EventHandler::new(button)
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    /// The AI picker popup: a card listing the on-device models (click to pick),
    /// the Turbo accelerated endpoint, and a disabled "Cloud — coming soon" slot
    /// (BYOK provider selection is on the roadmap). Returns `None` when closed.
    /// The model picker.
    ///
    /// Two questions, kept apart because conflating them is what made the old
    /// list confusing: *which model* answers (the list), and *how it runs* (the
    /// controls under the divider). Turbo used to sit in the list as though it
    /// were a model you could choose instead of Llama, when it is the backend
    /// the chosen model runs on.
    fn render_model_picker(&self, appearance: &Appearance) -> Option<Box<dyn Element>> {
        if !self.model_picker_open {
            return None;
        }
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();

        let note = |me: &Self, text: &str| -> Box<dyn Element> {
            Container::new(me.label_text(appearance, text, CHIP_FONT_SIZE, muted, true))
                .with_horizontal_padding(8.)
                .with_padding_top(4.)
                .with_padding_bottom(6.)
                .finish()
        };

        let mut list = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);

        // Where the model runs is the first choice, so it leads: everything
        // below it changes meaning depending on the answer.
        list.add_child(
            Container::new(
                Flex::row()
                    .with_child(self.picker_tab_button(appearance, "Local", ModelPickerTab::Local))
                    .with_child(self.picker_tab_button(appearance, "API", ModelPickerTab::Cloud))
                    .finish(),
            )
            .with_background_color(theme.surface_1().into())
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(9.)))
            .with_horizontal_padding(2.)
            .with_vertical_padding(2.)
            .with_margin_bottom(8.)
            .finish(),
        );

        match self.picker_tab {
            ModelPickerTab::Local => {
                if self.models.is_empty() {
                    list.add_child(note(
                        self,
                        "No local models yet. Pull one with ollama, or import a .gguf in the \
                         AI Mode Monitor, then Refresh.",
                    ));
                } else {
                    for (index, model) in self.models.iter().enumerate() {
                        // A GGUF runs on Turbo and an ollama tag on Ollama, and
                        // picking one sets the matching endpoint - so the endpoint
                        // no longer decides selection, the index does.
                        let selected = !self.cloud_active && self.selected_model == Some(index);
                        let loading = self.preparing_model.as_deref() == Some(model.as_str());
                        list.add_child(self.picker_model_row(
                            appearance,
                            self.model_label(model),
                            if loading { "loading" } else { "" },
                            LocalAiChatAction::PickModel(index),
                            selected,
                        ));
                    }
                }

                let gguf_active = self
                    .current_model()
                    .is_some_and(|model| is_gguf_ref(&model));
                let turbo_on = self.endpoint == LocalEndpoint::Turbo || gguf_active;

                list.add_child(Self::picker_divider(appearance));

                // Turbo below the divider, described by what it does to the model
                // above rather than offered as an alternative to it.
                list.add_child(self.picker_switch_row(
                    appearance,
                    "Turbo",
                    if gguf_active {
                        "Always on for a .gguf - the only backend that loads one"
                    } else if self.turbo_available {
                        "Run the selected model with full GPU offload"
                    } else {
                        "Not running - start genesi-ai-turbo to enable"
                    },
                    turbo_on,
                    LocalAiChatAction::ToggleTurbo,
                ));

                list.add_child(self.picker_switch_row(
                    appearance,
                    "AI Mode",
                    &ai_mode_short_label(self.ai_mode.as_ref()),
                    self.ai_mode.is_some(),
                    LocalAiChatAction::CycleAiMode,
                ));
            }
            ModelPickerTab::Cloud => {
                for provider in cloud_presets() {
                    let selected = self.cloud.provider == *provider;
                    list.add_child(self.picker_model_row(
                        appearance,
                        provider.label().to_string(),
                        "",
                        LocalAiChatAction::SelectCloudProvider(*provider),
                        self.cloud_active && selected,
                    ));
                }

                list.add_child(Self::picker_divider(appearance));

                for model in self.cloud.provider.suggested_models() {
                    let selected = self.cloud.model == *model;
                    list.add_child(self.picker_model_row(
                        appearance,
                        (*model).to_string(),
                        "",
                        LocalAiChatAction::PickCloudModel((*model).to_string()),
                        selected,
                    ));
                }

                let key_saved = !self.active_cloud_key().trim().is_empty();
                list.add_child(Self::picker_divider(appearance));
                list.add_child(self.picker_switch_row(
                    appearance,
                    "API key",
                    if key_saved {
                        "Stored on this machine"
                    } else {
                        "Needed before this provider can answer"
                    },
                    key_saved,
                    LocalAiChatAction::SetKey,
                ));
            }
        }

        let card = self.popup_surface(list.finish());
        let wrapped = if self.vibe_mode {
            ConstrainedBox::new(card)
                .with_width(VIBE_COLUMN_WIDTH)
                .finish()
        } else {
            card
        };
        Some(
            Container::new(wrapped)
                .with_horizontal_padding(PANEL_PADDING)
                .finish(),
        )
    }

    /// A hairline between "which model" and "how it runs".
    fn picker_divider(appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        Container::new(
            ConstrainedBox::new(
                Container::new(Empty::new().finish())
                    .with_background_color(theme.surface_2().into())
                    .finish(),
            )
            .with_height(1.)
            .finish(),
        )
        .with_margin_top(6.)
        .with_margin_bottom(6.)
        .finish()
    }

    /// A model in the list: name on the left, a live dot on the right.
    ///
    /// The dot sits in its own column and is drawn only for the active model,
    /// so the current choice is found by scanning one edge rather than by
    /// telling a filled circle from a hollow one at the head of every line.
    fn picker_model_row(
        &self,
        appearance: &Appearance,
        label: String,
        detail: &str,
        action: LocalAiChatAction,
        active: bool,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let text_color: ColorU = theme.main_text_color(theme.background()).into();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();

        let mut row = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_main_axis_size(MainAxisSize::Max)
            .with_child(self.label_text(appearance, &label, BODY_FONT_SIZE, text_color, false));

        if !detail.is_empty() {
            row.add_child(
                Container::new(self.label_text(appearance, detail, CHIP_FONT_SIZE, muted, false))
                    .with_margin_left(6.)
                    .finish(),
            );
        }

        row.add_child(Shrinkable::new(1., Container::new(Empty::new().finish()).finish()).finish());

        if active {
            row.add_child(
                ConstrainedBox::new(
                    Container::new(Empty::new().finish())
                        .with_background_color(genesi_green().into())
                        .with_corner_radius(CornerRadius::with_all(Radius::Percentage(50.)))
                        .finish(),
                )
                .with_width(7.)
                .with_height(7.)
                .finish(),
            );
        }

        let mut container = Container::new(row.finish())
            .with_horizontal_padding(8.)
            .with_vertical_padding(7.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)));
        if active {
            container = container.with_background_color(green_tint());
        }

        EventHandler::new(container.finish())
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    /// A control under the divider: what it does, and whether it is on.
    ///
    /// Each carries its own explanation, because these are the settings people
    /// guess wrong about - "Turbo" on its own says nothing about what enabling
    /// it needs or costs.
    fn picker_switch_row(
        &self,
        appearance: &Appearance,
        label: &str,
        detail: &str,
        on: bool,
        action: LocalAiChatAction,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let text_color: ColorU = if on {
            genesi_green()
        } else {
            theme.main_text_color(theme.background()).into()
        };
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();

        let mut heading = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_main_axis_size(MainAxisSize::Max)
            .with_child(self.label_text(appearance, label, BODY_FONT_SIZE, text_color, false));
        heading.add_child(
            Shrinkable::new(1., Container::new(Empty::new().finish()).finish()).finish(),
        );
        heading.add_child(self.label_text(
            appearance,
            if on { "On" } else { "Off" },
            CHIP_FONT_SIZE,
            if on { genesi_green() } else { muted },
            false,
        ));

        let column = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(heading.finish())
            .with_child(self.label_text(appearance, detail, CHIP_FONT_SIZE, muted, true))
            .finish();

        EventHandler::new(
            Container::new(column)
                .with_horizontal_padding(8.)
                .with_vertical_padding(6.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
                .finish(),
        )
        .on_left_mouse_down(move |ctx, _, _| {
            ctx.dispatch_typed_action(action.clone());
            DispatchEventResult::StopPropagation
        })
        .finish()
    }

    /// One half of the Local / Cloud segmented control at the top of the picker.
    fn picker_tab_button(
        &self,
        appearance: &Appearance,
        label: &str,
        tab: ModelPickerTab,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let selected = self.picker_tab == tab;
        let color: ColorU = if selected {
            genesi_green()
        } else {
            theme.sub_text_color(theme.background()).into()
        };
        let mut container =
            Container::new(self.label_text(appearance, label, CHIP_FONT_SIZE, color, false))
                .with_horizontal_padding(10.)
                .with_vertical_padding(5.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(7.)))
                .with_margin_right(4.);
        if selected {
            container = container.with_background_color(green_tint());
        }

        EventHandler::new(container.finish())
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(LocalAiChatAction::SetPickerTab(tab));
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    /// One clickable row in the model picker: a full-width hit target that tints
    /// green when it's the active selection.
    fn picker_row(
        &self,
        appearance: &Appearance,
        label: String,
        action: LocalAiChatAction,
        active: bool,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let color: ColorU = if active {
            genesi_green()
        } else {
            theme.main_text_color(theme.background()).into()
        };
        let mut row =
            Container::new(self.label_text(appearance, label, BODY_FONT_SIZE, color, false))
                .with_horizontal_padding(8.)
                .with_vertical_padding(6.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)));
        if active {
            row = row.with_background_color(green_tint());
        }

        EventHandler::new(row.finish())
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }

    /// The review bar: a summary of the file changes the agent has made this
    /// turn, with "Undo all" / "Keep" — like the reference's "N files need
    /// review". Sits just above the compose box; `None` when nothing's pending.
    fn render_review_bar(&self, appearance: &Appearance) -> Option<Box<dyn Element>> {
        if self.pending_edits.is_empty() {
            return None;
        }
        let theme = appearance.theme();
        let n = self.pending_edits.len();
        let added: u32 = self.pending_edits.iter().map(|e| e.added).sum();
        let removed: u32 = self.pending_edits.iter().map(|e| e.removed).sum();
        let label = format!("> {n} file{} need review", if n == 1 { "" } else { "s" });
        let close_icon_color: ColorU = theme.sub_text_color(theme.background()).into();

        let row = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_main_axis_size(MainAxisSize::Max)
            .with_child(self.label_text(
                appearance,
                label,
                CHIP_FONT_SIZE,
                theme.main_text_color(theme.background()).into(),
                false,
            ))
            .with_child(
                Container::new(self.label_text(
                    appearance,
                    format!("+{added}"),
                    CHIP_FONT_SIZE,
                    genesi_green(),
                    false,
                ))
                .with_margin_left(10.)
                .with_margin_right(4.)
                .finish(),
            )
            .with_child(self.label_text(
                appearance,
                format!("-{removed}"),
                CHIP_FONT_SIZE,
                theme.ui_error_color().into(),
                false,
            ))
            .with_child(Shrinkable::new(1., Empty::new().finish()).finish())
            .with_child(self.chip(
                appearance,
                "Undo".to_string(),
                LocalAiChatAction::UndoEdits,
                false,
            ))
            .with_child(self.chip(
                appearance,
                "Keep All".to_string(),
                LocalAiChatAction::KeepEdits,
                true,
            ))
            // A way OUT of the bar. Closing it means the edits stay on disk — the
            // same outcome as Keep All, which is why it dispatches that rather
            // than inventing a third, vaguer state.
            .with_child(
                EventHandler::new(
                    Container::new(
                        ConstrainedBox::new(
                            Icon::new("bundled/svg/x-close.svg", close_icon_color).finish(),
                        )
                        .with_width(11.)
                        .with_height(11.)
                        .finish(),
                    )
                    .with_uniform_padding(5.)
                    .with_margin_left(2.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
                    .finish(),
                )
                .on_left_mouse_down(move |ctx, _, _| {
                    ctx.dispatch_typed_action(LocalAiChatAction::KeepEdits);
                    DispatchEventResult::StopPropagation
                })
                .finish(),
            )
            .finish();

        let column = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(
                EventHandler::new(
                    Container::new(row)
                        .with_uniform_padding(8.)
                        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                        .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
                        .with_background_color(genesi_panel_surface())
                        .finish(),
                )
                .on_left_mouse_down(move |ctx, _, _| {
                    ctx.dispatch_typed_action(LocalAiChatAction::ToggleReviewExpanded);
                    DispatchEventResult::StopPropagation
                })
                .finish(),
            );

        let wrapped = if self.vibe_mode {
            ConstrainedBox::new(column.finish())
                .with_width(VIBE_COLUMN_WIDTH)
                .finish()
        } else {
            column.finish()
        };
        Some(
            Container::new(wrapped)
                .with_horizontal_padding(PANEL_PADDING)
                .with_padding_bottom(4.)
                .finish(),
        )
    }

    /// The approval prompt shown while a side-effecting tool waits for the user.
    fn render_approval(&self, appearance: &Appearance, tool: &AgentTool) -> Box<dyn Element> {
        let theme = appearance.theme();
        let clip = |s: &str| -> String {
            if s.chars().count() > 240 {
                s.chars().take(240).collect::<String>() + "…"
            } else {
                s.to_string()
            }
        };
        let (title, detail) = match tool {
            AgentTool::RunCommand { command } => {
                ("⚠ Allow this command?".to_string(), command.clone())
            }
            AgentTool::EditFile {
                path,
                search,
                replace,
            } => (
                format!("✏ Apply this edit to {path}?"),
                format!("- {}\n+ {}", clip(search), clip(replace)),
            ),
            AgentTool::WriteFile { path, content } => (format!("✏ Write {path}?"), clip(content)),
            other => ("⚠ Allow this action?".to_string(), other.summary()),
        };

        let detail_box = Container::new(self.mono_text(
            appearance,
            detail,
            MONO_FONT_SIZE,
            theme.main_text_color(theme.background()).into(),
            true,
        ))
        .with_uniform_padding(8.)
        .with_margin_top(4.)
        .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
        .with_border(Border::all(1.).with_border_fill(theme.outline()))
        .with_background(theme.surface_1())
        .finish();

        let buttons = Flex::row()
            .with_cross_axis_alignment(CrossAxisAlignment::Center)
            .with_main_axis_size(MainAxisSize::Max)
            .with_child(self.chip(
                appearance,
                "Allow".to_string(),
                LocalAiChatAction::ApproveTool,
                true,
            ))
            .with_child(self.chip(
                appearance,
                "Deny".to_string(),
                LocalAiChatAction::DenyTool,
                true,
            ))
            .finish();

        let content = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(self.label_text(appearance, title, CHIP_FONT_SIZE, genesi_green(), false))
            .with_child(detail_box)
            .with_child(Container::new(buttons).with_margin_top(6.).finish())
            .finish();
        let wrapped = if self.vibe_mode {
            ConstrainedBox::new(content)
                .with_width(VIBE_COLUMN_WIDTH)
                .finish()
        } else {
            content
        };
        Container::new(wrapped)
            .with_horizontal_padding(PANEL_PADDING)
            .with_padding_bottom(6.)
            .finish()
    }

    /// A clickable, collapse/expand header for the thought/tool/command steps.
    fn step_header(
        &self,
        appearance: &Appearance,
        index: usize,
        title: String,
        color: ColorU,
        collapsed: bool,
    ) -> Box<dyn Element> {
        self.collapsible_header(
            appearance,
            LocalAiChatAction::ToggleCollapse(index),
            title,
            color,
            collapsed,
        )
    }

    /// A disclosure row: chevron + title, toggling `action` when clicked.
    fn collapsible_header(
        &self,
        appearance: &Appearance,
        action: LocalAiChatAction,
        title: String,
        color: ColorU,
        collapsed: bool,
    ) -> Box<dyn Element> {
        // A real chevron, not the literal ">" / "v" characters this used to
        // print — those rendered as stray letters in the middle of the
        // transcript.
        let caret = if collapsed {
            "bundled/svg/chevron-right-skinny.svg"
        } else {
            "bundled/svg/chevron-down-skinny.svg"
        };
        let row = Container::new(
            Flex::row()
                .with_cross_axis_alignment(CrossAxisAlignment::Center)
                .with_child(
                    Container::new(
                        ConstrainedBox::new(Icon::new(caret, color).finish())
                            .with_width(10.)
                            .with_height(10.)
                            .finish(),
                    )
                    .with_margin_right(5.)
                    .finish(),
                )
                .with_child(self.label_text(appearance, title, CHIP_FONT_SIZE, color, false))
                .finish(),
        )
        .with_vertical_padding(2.)
        .finish();

        EventHandler::new(row)
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(action.clone());
                DispatchEventResult::StopPropagation
            })
            .finish()
    }
    /// A user or assistant message bubble (assistant text rendered as markdown).
    fn render_bubble(&self, appearance: &Appearance, entry: &ChatEntry) -> Box<dyn Element> {
        let theme = appearance.theme();
        let is_user = entry.role == ChatRole::User;
        let (prefix, role_color): (&str, ColorU) = if is_user {
            ("You", theme.disabled_text_color(theme.background()).into())
        } else {
            (
                "Genesi AI",
                theme.disabled_text_color(theme.background()).into(),
            )
        };

        let mut inner = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(self.label_text(appearance, prefix, CHIP_FONT_SIZE, role_color, false));

        if let Some(label) = &entry.context_label {
            let pill = Container::new(self.label_text(
                appearance,
                format!("Attach: {label}"),
                CHIP_FONT_SIZE,
                theme.disabled_text_color(theme.background()).into(),
                false,
            ))
            .with_horizontal_padding(6.)
            .with_vertical_padding(2.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(4.)))
            .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
            .with_background_color(ColorU::new(0, 0, 0, 0))
            .finish();

            inner.add_child(
                Container::new(
                    Flex::row()
                        .with_main_axis_size(MainAxisSize::Max)
                        .with_child(pill)
                        .with_child(Shrinkable::new(1., Empty::new().finish()).finish())
                        .finish(),
                )
                .with_margin_top(4.)
                .finish(),
            );
        }

        let body_color = theme.main_text_color(theme.background()).into();
        let body_el: Box<dyn Element> = if entry.text.trim().is_empty() {
            let dots = if self.in_flight { "..." } else { "" };
            self.label_text(appearance, dots, BODY_FONT_SIZE, body_color, true)
        } else if entry.role == ChatRole::Assistant {
            self.markdown_text(appearance, &entry.text, body_color)
        } else {
            self.label_text(
                appearance,
                entry.text.clone(),
                BODY_FONT_SIZE,
                body_color,
                true,
            )
        };

        let inner = inner
            .with_child(Container::new(body_el).with_margin_top(4.).finish())
            .finish();

        let bubble = Container::new(inner)
            .with_margin_bottom(10.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)));
        let bubble = if is_user {
            bubble
                .with_uniform_padding(10.)
                .with_background_color(genesi_card_surface())
                .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
        } else {
            bubble.with_vertical_padding(2.)
        };
        bubble.finish()
    }
    /// A collapsible thought, kept out of the way so the transcript stays clean.
    fn render_thought(
        &self,
        appearance: &Appearance,
        index: usize,
        entry: &ChatEntry,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let title = if entry.status == StepStatus::Running {
            "Thinking...".to_string()
        } else {
            "Thought".to_string()
        };

        let mut column = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(self.step_header(appearance, index, title, muted, entry.collapsed));

        if !entry.collapsed && !entry.text.trim().is_empty() {
            column.add_child(
                Container::new(self.label_text(
                    appearance,
                    entry.text.clone(),
                    BODY_FONT_SIZE,
                    muted,
                    true,
                ))
                .with_horizontal_padding(10.)
                .with_margin_top(2.)
                .finish(),
            );
        }

        Container::new(column.finish())
            .with_margin_bottom(6.)
            .finish()
    }
    /// A collapsible read-tool step: a one-line summary with the (previewed)
    /// result one click away.
    fn render_tool_step(
        &self,
        appearance: &Appearance,
        index: usize,
        entry: &ChatEntry,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let (icon, color): (&str, ColorU) = match entry.status {
            StepStatus::Running => ("file", muted),
            StepStatus::Ok => ("file", theme.main_text_color(theme.background()).into()),
            StepStatus::Error => ("error", theme.ui_error_color().into()),
            StepStatus::Denied => ("denied", muted),
        };
        let suffix = match entry.status {
            StepStatus::Running => " - running...",
            StepStatus::Error => " - error",
            StepStatus::Denied => " - denied",
            StepStatus::Ok => "",
        };
        let path = entry.tool_title.clone().unwrap_or_default();
        let header: Box<dyn Element> = if let Some((added, removed)) = entry.diff_stat {
            let caret = if entry.collapsed { ">" } else { "v" };
            let green: ColorU = genesi_green();
            let red: ColorU = theme.ui_error_color().into();
            let row = Flex::row()
                .with_cross_axis_alignment(CrossAxisAlignment::Center)
                .with_main_axis_size(MainAxisSize::Max)
                .with_child(self.label_text(
                    appearance,
                    format!("{caret} {path}{suffix}"),
                    CHIP_FONT_SIZE,
                    color,
                    false,
                ))
                .with_child(Shrinkable::new(1., Empty::new().finish()).finish())
                .with_child(
                    Container::new(self.label_text(
                        appearance,
                        format!("+{added}"),
                        CHIP_FONT_SIZE,
                        green,
                        false,
                    ))
                    .with_margin_right(4.)
                    .finish(),
                )
                .with_child(self.label_text(
                    appearance,
                    format!("-{removed}"),
                    CHIP_FONT_SIZE,
                    red,
                    false,
                ))
                .with_child(
                    Container::new(self.chip(
                        appearance,
                        "Open Diff".to_string(),
                        LocalAiChatAction::OpenDiff(index),
                        false,
                    ))
                    .with_margin_left(8.)
                    .finish(),
                )
                .finish();
            EventHandler::new(
                Container::new(row)
                    .with_horizontal_padding(8.)
                    .with_vertical_padding(6.)
                    .with_corner_radius(CornerRadius::with_all(Radius::Pixels(8.)))
                    .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
                    .with_background_color(genesi_card_surface())
                    .finish(),
            )
            .on_left_mouse_down(move |ctx, _, _| {
                ctx.dispatch_typed_action(LocalAiChatAction::ToggleCollapse(index));
                DispatchEventResult::StopPropagation
            })
            .finish()
        } else {
            let title = format!("{icon}: {path}{suffix}");
            self.step_header(appearance, index, title, color, entry.collapsed)
        };

        let mut column = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(header);

        if !entry.collapsed {
            if entry.diff_preview.is_none() && !entry.text.trim().is_empty() {
                let body = Container::new(self.mono_text(
                    appearance,
                    entry.text.clone(),
                    MONO_FONT_SIZE,
                    theme.main_text_color(theme.background()).into(),
                    true,
                ))
                .with_uniform_padding(8.)
                .with_margin_top(2.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
                .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
                .with_background_color(genesi_card_surface())
                .finish();
                column.add_child(body);
            }
        }

        Container::new(column.finish())
            .with_margin_bottom(6.)
            .finish()
    }
    /// A `run_command` step, rendered like the app's integrated terminal: a dark
    /// block with a green `$ command` prompt and the captured output beneath.
    fn render_command_step(
        &self,
        appearance: &Appearance,
        index: usize,
        entry: &ChatEntry,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let green: ColorU = theme.terminal_colors().normal.green.into();
        let suffix = match entry.status {
            StepStatus::Running => " · running…",
            StepStatus::Error => " · exited with error",
            StepStatus::Denied => " · stopped",
            StepStatus::Ok => "",
        };
        let header_color: ColorU = if entry.status == StepStatus::Error {
            theme.ui_error_color().into()
        } else {
            green
        };

        let mut column = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(self.step_header(
                appearance,
                index,
                format!("⌘ Terminal{suffix}"),
                header_color,
                entry.collapsed,
            ));

        if !entry.collapsed {
            let command = entry.command.clone().unwrap_or_default();
            let mut terminal = Flex::column()
                .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                .with_child(self.mono_text(
                    appearance,
                    format!("$ {command}"),
                    MONO_FONT_SIZE,
                    green,
                    true,
                ));
            let body = if entry.text.trim().is_empty() && entry.status == StepStatus::Running {
                "running…".to_string()
            } else {
                entry.text.clone()
            };
            if !body.trim().is_empty() {
                terminal.add_child(
                    Container::new(self.mono_text(
                        appearance,
                        body,
                        MONO_FONT_SIZE,
                        theme.main_text_color(theme.background()).into(),
                        true,
                    ))
                    .with_margin_top(4.)
                    .finish(),
                );
            }
            let block = Container::new(terminal.finish())
                .with_uniform_padding(8.)
                .with_margin_top(2.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(6.)))
                .with_border(Border::all(1.).with_border_fill(theme.outline()))
                .with_background(theme.background())
                .finish();
            column.add_child(block);
        }

        Container::new(column.finish())
            .with_margin_bottom(6.)
            .finish()
    }

    /// The extent of a FINISHED run of agent steps starting at `start`, as
    /// `(end_exclusive, tool_step_count)`.
    ///
    /// Returns `None` unless the run holds at least two tool steps — a lone step
    /// reads better on its own than behind a disclosure. A step still running is
    /// never swallowed: the user needs to see what the agent is doing right now.
    fn finished_step_run(&self, start: usize) -> Option<(usize, usize)> {
        let mut end = start;
        let mut steps = 0usize;
        while let Some(entry) = self.messages.get(end) {
            if entry.status == StepStatus::Running {
                break;
            }
            match entry.role {
                ChatRole::Tool | ChatRole::Command => {
                    steps += 1;
                    end += 1;
                }
                ChatRole::Thought => end += 1,
                _ => break,
            }
        }
        // Trailing thoughts with no tool after them read as the model's closing
        // reasoning; leave them out of the group.
        while end > start && steps > 0 && self.messages[end - 1].role == ChatRole::Thought {
            end -= 1;
        }
        (steps >= 2).then_some((end, steps))
    }

    /// Summarise what a run of steps actually DID, by kind.
    ///
    /// Calling everything "Ran N commands" was wrong: reading a file, writing
    /// one and running a shell command are three different things, and lumping
    /// them together hid whether the agent had touched the disk at all. The
    /// kinds are told apart by what the transcript already records — a Command
    /// entry is a shell command, and a Tool entry carrying a diff wrote a file.
    fn step_run_title(&self, start: usize, end: usize, steps: usize) -> String {
        let (mut commands, mut writes, mut reads) = (0usize, 0usize, 0usize);
        let mut failed = 0usize;
        for entry in &self.messages[start..end] {
            // A step that errored changed nothing, so it must not be counted as
            // work done — say so instead.
            if entry.status == StepStatus::Error {
                if matches!(entry.role, ChatRole::Tool | ChatRole::Command) {
                    failed += 1;
                }
                continue;
            }
            match entry.role {
                ChatRole::Command => commands += 1,
                ChatRole::Tool if entry.diff_stat.is_some() => writes += 1,
                ChatRole::Tool => reads += 1,
                _ => {}
            }
        }
        fn plural<'a>(n: usize, one: &'a str, many: &'a str) -> &'a str {
            if n == 1 {
                one
            } else {
                many
            }
        }
        let mut parts = Vec::new();
        if commands > 0 {
            parts.push(format!(
                "Ran {commands} {}",
                plural(commands, "command", "commands")
            ));
        }
        if writes > 0 {
            parts.push(format!(
                "Edited {writes} {}",
                plural(writes, "file", "files")
            ));
        }
        if reads > 0 {
            parts.push(format!("Read {reads} {}", plural(reads, "file", "files")));
        }
        if failed > 0 {
            parts.push(format!("{failed} failed"));
        }
        if parts.is_empty() {
            return format!("{steps} steps");
        }
        parts.join(" · ")
    }

    /// One row standing in for a run of tool steps: "Ran 5 commands", expanding
    /// to the individual steps.
    fn render_step_group(
        &self,
        appearance: &Appearance,
        start: usize,
        end: usize,
        steps: usize,
    ) -> Box<dyn Element> {
        let theme = appearance.theme();
        let muted: ColorU = theme.disabled_text_color(theme.background()).into();
        let expanded = self.expanded_step_groups.contains(&start);
        let title = self.step_run_title(start, end, steps);

        let mut column = Flex::column()
            .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
            .with_child(self.collapsible_header(
                appearance,
                LocalAiChatAction::ToggleStepGroup(start),
                title,
                muted,
                !expanded,
            ));

        if expanded {
            for index in start..end {
                column.add_child(
                    Container::new(self.render_entry(appearance, index, &self.messages[index]))
                        .with_padding_left(12.)
                        .finish(),
                );
            }
        }

        Container::new(column.finish())
            .with_margin_bottom(6.)
            .finish()
    }

    fn render_entry(
        &self,
        appearance: &Appearance,
        index: usize,
        entry: &ChatEntry,
    ) -> Box<dyn Element> {
        match entry.role {
            ChatRole::User | ChatRole::Assistant => self.render_bubble(appearance, entry),
            ChatRole::Thought => self.render_thought(appearance, index, entry),
            ChatRole::Tool => self.render_tool_step(appearance, index, entry),
            ChatRole::Command => self.render_command_step(appearance, index, entry),
        }
    }

    /// The three capability tiles under the vibe-mode hero. Informational only —
    /// they describe what this assistant actually does here, so nothing is
    /// promised that the panel can't deliver.
    /// The zero-state cards.
    ///
    /// Each is a way in, not a brochure entry: clicking one seeds the compose
    /// box with the prompt underneath it. The footer line names the surface the
    /// card leans on, so the card says what it will actually do.
    fn render_vibe_capability_cards(&self, appearance: &Appearance) -> Box<dyn Element> {
        const CARDS: [(&str, &str, &str, &str, &str); 3] = [
            (
                "bundled/svg/search.svg",
                "Understand this project",
                "Reads files on demand and explains how they fit together.",
                "Explain how this project is structured and where the entry points are.",
                "Reads your files",
            ),
            (
                "bundled/svg/agentmode.svg",
                "Write and edit",
                "Proposes changes as a diff you review before anything lands.",
                "Find the rough edges in the file I have open and propose fixes.",
                "Diff before it lands",
            ),
            (
                "bundled/svg/lightning-02.svg",
                "Runs on your machine",
                "Local models by default - no account and no cloud required.",
                "Which of my local models is the best fit for this codebase?",
                "Local models",
            ),
        ];

        let theme = appearance.theme();
        let title_color: ColorU = theme.active_ui_text_color().into();
        let body_color: ColorU = theme.disabled_text_color(theme.background()).into();
        let footer_color: ColorU = theme.nonactive_ui_text_color().into();

        let mut row = Flex::row().with_cross_axis_alignment(CrossAxisAlignment::Stretch);
        for (index, (icon, title, body, prompt, footer)) in CARDS.iter().enumerate() {
            let card = Container::new(
                Flex::column()
                    .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                    .with_child(
                        ConstrainedBox::new(Icon::new(icon, genesi_green()).finish())
                            .with_width(16.)
                            .with_height(16.)
                            .finish(),
                    )
                    .with_child(
                        Container::new(self.label_text(
                            appearance,
                            (*title).to_string(),
                            BODY_FONT_SIZE,
                            title_color,
                            false,
                        ))
                        .with_margin_top(10.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(self.label_text(
                            appearance,
                            (*body).to_string(),
                            CHIP_FONT_SIZE,
                            body_color,
                            true,
                        ))
                        .with_margin_top(4.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(self.label_text(
                            appearance,
                            (*footer).to_string(),
                            CHIP_FONT_SIZE - 1.,
                            footer_color,
                            false,
                        ))
                        .with_margin_top(14.)
                        .finish(),
                    )
                    .finish(),
            )
            .with_uniform_padding(14.)
            .with_corner_radius(CornerRadius::with_all(Radius::Pixels(12.)))
            .with_background_color(genesi_card_surface())
            .with_border(Border::all(1.).with_border_color(genesi_subtle_border()))
            .with_margin_right(if index + 1 < CARDS.len() { 10. } else { 0. })
            .finish();

            let clickable = EventHandler::new(card)
                .on_left_mouse_down(move |ctx, _, _| {
                    ctx.dispatch_typed_action(LocalAiChatAction::StartPrompt(prompt));
                    DispatchEventResult::StopPropagation
                })
                .finish();
            row.add_child(Expanded::new(1., clickable).finish());
        }
        row.finish()
    }

    /// The quick row under the cards: the handful of things people reach for
    /// before they have typed anything.
    fn render_vibe_quick_actions(&self, appearance: &Appearance) -> Box<dyn Element> {
        let mut row = Flex::row()
            .with_main_axis_alignment(MainAxisAlignment::Center)
            .with_cross_axis_alignment(CrossAxisAlignment::Center);

        for (label, icon, action) in [
            (
                "Review my changes",
                "bundled/svg/git-branch-02.svg",
                LocalAiChatAction::StartPrompt(
                    "Review the changes I have not committed yet and tell me what looks wrong.",
                ),
            ),
            (
                "Write a test",
                "bundled/svg/check.svg",
                LocalAiChatAction::StartPrompt(
                    "Write a test for the code I have open, covering the cases it gets wrong.",
                ),
            ),
            (
                "Attach a file",
                "bundled/svg/paperclip.svg",
                LocalAiChatAction::PickAttachments,
            ),
            (
                "Models",
                "bundled/svg/layers-three-01.svg",
                LocalAiChatAction::ToggleModelPicker,
            ),
        ] {
            row.add_child(
                Container::new(self.chip_with_icon(
                    appearance,
                    label.to_string(),
                    Some(icon),
                    action,
                    false,
                    true,
                ))
                .with_margin_right(8.)
                .finish(),
            );
        }

        row.finish()
    }

    fn render_transcript(&self, appearance: &Appearance) -> Box<dyn Element> {
        let theme = appearance.theme();
        let mut column = Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);

        if self.messages.is_empty() {
            if self.vibe_mode {
                let hero = Flex::column()
                    .with_main_axis_alignment(MainAxisAlignment::Center)
                    .with_cross_axis_alignment(CrossAxisAlignment::Center)
                    .with_child(
                        Container::new(
                            Container::new(
                                ConstrainedBox::new(
                                    Icon::new("bundled/svg/sparkle.svg", genesi_green()).finish(),
                                )
                                .with_width(34.)
                                .with_height(34.)
                                .finish(),
                            )
                            .with_uniform_padding(14.)
                            .with_corner_radius(CornerRadius::with_all(Radius::Percentage(50.)))
                            .with_background_color(ColorU::new(15, 143, 106, 46))
                            .with_border(
                                Border::all(1.).with_border_color(ColorU::new(15, 143, 106, 90)),
                            )
                            .finish(),
                        )
                        .with_uniform_padding(7.)
                        .with_corner_radius(CornerRadius::with_all(Radius::Percentage(50.)))
                        .with_background_color(ColorU::new(15, 143, 106, 18))
                        .finish(),
                    )
                    .with_child(
                        Container::new(self.label_text(
                            appearance,
                            "Ready to create something new?".to_string(),
                            23.,
                            theme.active_ui_text_color().into(),
                            false,
                        ))
                        .with_margin_top(18.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(
                            self.label_text(
                                appearance,
                                "Genesi Code can plan, write, review, and explain your project."
                                    .to_string(),
                                BODY_FONT_SIZE,
                                theme.disabled_text_color(theme.background()).into(),
                                false,
                            ),
                        )
                        .with_margin_top(8.)
                        .finish(),
                    )
                    .with_child(
                        Container::new(self.render_vibe_capability_cards(appearance))
                            .with_margin_top(26.)
                            .finish(),
                    )
                    .with_child(
                        Container::new(self.render_vibe_quick_actions(appearance))
                            .with_margin_top(16.)
                            .finish(),
                    )
                    .finish();
                return Container::new(hero)
                    .with_horizontal_padding(PANEL_PADDING)
                    .finish();
            }
            let (title, hint) = if self.cloud_active {
                match self.current_model() {
                    Some(model) => (
                        format!("Ask {}", self.cloud.provider.label()),
                        format!(
                            "Using {model} via {}. Your API key stays on this device in secure storage. With Attach on, the file you're editing is sent as context.",
                            self.cloud.provider.label()
                        ),
                    ),
                    None => (
                        format!("Set up {}", self.cloud.provider.label()),
                        format!(
                            "Pick a model and save an API key for {} to start chatting.",
                            self.cloud.provider.label()
                        ),
                    ),
                }
            } else {
                match self.current_model() {
                    Some(model) => (
                        "Ask your local model".to_string(),
                        format!(
                            "Running {model} on-device — no account, no cloud. With 📎 on, \
                             the file you're editing is sent as context."
                        ),
                    ),
                    None => (
                        "No local model yet".to_string(),
                        "Start ollama and pull a model (e.g. `ollama pull llama3.2`), then hit Refresh."
                            .to_string(),
                    ),
                }
            };
            column.add_child(
                Container::new(
                    Flex::column()
                        .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                        .with_child(self.label_text(
                            appearance,
                            title,
                            TITLE_FONT_SIZE,
                            theme.active_ui_text_color().into(),
                            false,
                        ))
                        .with_child(
                            Container::new(self.label_text(
                                appearance,
                                hint,
                                BODY_FONT_SIZE,
                                theme.disabled_text_color(theme.background()).into(),
                                true,
                            ))
                            .with_margin_top(6.)
                            .finish(),
                        )
                        .finish(),
                )
                .with_uniform_padding(12.)
                .with_margin_top(8.)
                .finish(),
            );
        } else {
            let mut index = 0;
            while index < self.messages.len() {
                // A finished run of tool steps collapses into one "Ran N
                // commands" row. Interleaved thoughts belong to that run, so
                // they fold in too — otherwise a multi-step task buries the
                // conversation under dozens of one-line steps.
                if let Some((end, steps)) = self.finished_step_run(index) {
                    column.add_child(self.render_step_group(appearance, index, end, steps));
                    index = end;
                    continue;
                }
                column.add_child(self.render_entry(appearance, index, &self.messages[index]));
                index += 1;
            }
        }

        let selected_text = self.selected_transcript_text.clone();
        let selectable_transcript = SelectableArea::new(
            self.transcript_selection.clone(),
            move |selection, _, _| {
                *selected_text.write() = selection.selection.filter(|text| !text.is_empty());
            },
            Container::new(column.finish())
                .with_horizontal_padding(PANEL_PADDING)
                .finish(),
        )
        .on_selection_updated(|ctx, _| {
            ctx.dispatch_typed_action(LocalAiChatAction::FocusTranscript);
        })
        .finish();

        ClippedScrollable::vertical(
            self.transcript_scroll.clone(),
            selectable_transcript,
            ScrollbarWidth::Auto,
            theme.disabled_ui_text_color().into(),
            theme.active_ui_text_color().into(),
            Fill::None,
        )
        .finish()
    }
}

impl Entity for LocalAiChatView {
    type Event = LocalAiChatEvent;
}

impl TypedActionView for LocalAiChatView {
    type Action = LocalAiChatAction;

    fn handle_action(&mut self, action: &LocalAiChatAction, ctx: &mut ViewContext<Self>) {
        match action {
            LocalAiChatAction::CopySelectedText => {
                if let Some(text) = self
                    .selected_transcript_text
                    .read()
                    .clone()
                    .filter(|text| !text.is_empty())
                {
                    ctx.clipboard().write(ClipboardContent::plain_text(text));
                } else {
                    self.input.update(ctx, |input, ctx| {
                        input.editor().update(ctx, |editor, ctx| editor.copy(ctx));
                    });
                }
            }
            LocalAiChatAction::FocusTranscript => {
                ctx.focus_self();
            }
            LocalAiChatAction::StartPrompt(prompt) => {
                let prompt = *prompt;
                self.input.update(ctx, |input, ctx| {
                    input.insert_pasted_text(prompt, ctx);
                });
                ctx.focus(&self.input);
                ctx.notify();
            }
            LocalAiChatAction::CycleEndpoint => {
                self.load_cloud_keys(ctx);
                // A deliberate choice — stop the Turbo auto-default from
                // overriding it on the next probe. Cycle: Local -> Turbo (if up)
                // -> Cloud (BYOK) -> Local.
                self.endpoint_user_chosen = true;
                self.error = None;
                if self.cloud_active {
                    self.cloud_active = false;
                    self.endpoint = LocalEndpoint::Ollama;
                    self.refresh_models(ctx);
                } else if self.endpoint == LocalEndpoint::Ollama && self.turbo_available {
                    self.endpoint = LocalEndpoint::Turbo;
                    self.refresh_models(ctx);
                } else {
                    // From Turbo, or from Local when Turbo is down -> Cloud.
                    // Reset the local endpoint so leaving cloud returns to Local.
                    self.endpoint = LocalEndpoint::Ollama;
                    self.cloud_active = true;
                    if !self.cloud_ready() {
                        self.error = Some(format!(
                            "Pick a provider and add your key (🔑 Set key) to use {}.",
                            self.cloud_label()
                        ));
                    }
                }
                ctx.notify();
            }
            LocalAiChatAction::CycleProvider => {
                self.load_cloud_keys(ctx);
                let presets = cloud_presets();
                let current = presets
                    .iter()
                    .position(|provider| *provider == self.cloud.provider)
                    .unwrap_or(0);
                let next = (current + 1) % presets.len();
                // Replace the model only if it was empty or still the old preset's
                // default — never clobber a model the user typed by hand.
                let previous_provider = presets[current];
                let replace_model = self.cloud.model.trim().is_empty()
                    || self.cloud.model == previous_provider.default_model();
                self.cloud.provider = presets[next];
                if replace_model {
                    self.cloud.model = self.cloud.provider.default_model().to_string();
                }
                if let Err(err) = save_cloud_config(&self.cloud) {
                    self.error = Some(format!("Couldn't save cloud config: {err}"));
                } else if !self.cloud_ready() {
                    self.error = Some(format!(
                        "Add your {} API key to use {}.",
                        self.cloud.provider.label(),
                        self.cloud.model
                    ));
                } else {
                    self.error = None;
                }
                ctx.notify();
            }
            LocalAiChatAction::SelectCloudProvider(provider) => {
                self.select_cloud_provider(*provider, false, ctx);
                ctx.notify();
            }
            LocalAiChatAction::PickCloudModel(model) => {
                self.load_cloud_keys(ctx);
                self.cloud_active = true;
                self.cloud.model = model.clone();
                match save_cloud_config(&self.cloud) {
                    Ok(()) => self.error = None,
                    Err(err) => self.error = Some(format!("Couldn't save cloud config: {err}")),
                }
                ctx.notify();
            }
            LocalAiChatAction::SetKey => ctx.dispatch_typed_action(
                &WorkspaceAction::ShowSettingsPage(SettingsSection::WarpAgent),
            ),
            LocalAiChatAction::SetModel => self.set_input_mode(InputMode::CloudModel, ctx),
            LocalAiChatAction::CycleModel => {
                if !self.models.is_empty() {
                    let next = self
                        .selected_model
                        .map(|index| (index + 1) % self.models.len())
                        .unwrap_or(0);
                    self.selected_model = Some(next);
                    // Cycling onto a GGUF must load it too, same as picking one.
                    if let Some(model) = self.models.get(next).cloned() {
                        self.ensure_model_ready(model, ctx);
                    }
                }
                ctx.notify();
            }
            LocalAiChatAction::CycleAiMode => {
                let current = self
                    .ai_mode
                    .as_ref()
                    .map(|state| state.force_mode.as_str())
                    .unwrap_or("auto");
                let next = match current {
                    "auto" => "on",
                    "on" => "off",
                    _ => "auto",
                };
                match set_ai_mode_force(next) {
                    Ok(()) => {
                        self.refresh_ai_mode();
                        self.error = None;
                    }
                    Err(e) => self.error = Some(format!("Couldn't change AI Mode: {e}")),
                }
                ctx.notify();
            }
            LocalAiChatAction::Refresh => {
                self.load_cloud_keys(ctx);
                self.refresh_models(ctx);
                self.refresh_ai_mode();
                ctx.notify();
            }
            LocalAiChatAction::Clear => {
                self.reset_current_chat();
                self.persist_mempalace(ctx, true);
                ctx.notify();
            }
            LocalAiChatAction::ToggleAttachContext => {
                self.attach_context = !self.attach_context;
                ctx.notify();
            }
            LocalAiChatAction::PickAttachments => self.pick_attachments(ctx),
            LocalAiChatAction::AttachPaths(paths) => {
                self.attach_paths(paths.clone().into_iter(), ctx)
            }
            LocalAiChatAction::RemoveAttachment(index) => {
                let index = *index;
                if index < self.attachments.len() {
                    self.attachments.remove(index);
                    ctx.notify();
                }
            }
            LocalAiChatAction::ToggleAgent => {
                self.agent_mode = !self.agent_mode;
                ctx.notify();
            }
            LocalAiChatAction::ToggleAuto => {
                self.auto_approve = !self.auto_approve;
                ctx.notify();
            }
            LocalAiChatAction::ToggleModelPicker => {
                self.load_cloud_keys(ctx);
                self.model_picker_open = !self.model_picker_open;
                // Only one popup at a time — they occupy the same slot above the
                // compose box and would stack awkwardly.
                self.mode_picker_open = false;
                ctx.notify();
            }
            LocalAiChatAction::ToggleModePicker => {
                self.mode_picker_open = !self.mode_picker_open;
                self.model_picker_open = false;
                ctx.notify();
            }
            LocalAiChatAction::SetChatMode(mode) => {
                self.set_chat_mode(*mode);
                self.mode_picker_open = false;
                // Remember it: resetting to the default on every launch is how
                // someone ends up back in an agent loop they had opted out of.
                self.persist_mempalace(ctx, false);
                ctx.notify();
            }
            LocalAiChatAction::SetPickerTab(tab) => {
                self.picker_tab = *tab;
                ctx.notify();
            }
            LocalAiChatAction::PickModel(index) => {
                // A deliberate local-model choice: pin the endpoint and stop the
                // Turbo auto-default from overriding it.
                self.endpoint_user_chosen = true;
                self.cloud_active = false;
                let picked = self.models.get(*index).cloned();
                if *index < self.models.len() {
                    self.selected_model = Some(*index);
                }
                self.model_picker_open = false;
                self.error = None;
                match picked {
                    // A GGUF can only be served by llama-server, so pinning
                    // ollama here would guarantee an empty reply. Switch to
                    // Turbo and load the file.
                    Some(model) if is_gguf_ref(&model) => self.ensure_model_ready(model, ctx),
                    // An ollama tag runs on EITHER transport. Forcing ollama here
                    // silently kicked the user off Turbo every time they picked
                    // one; keep the endpoint they chose, and if that's Turbo, tell
                    // it which model to load.
                    Some(model) if self.endpoint == LocalEndpoint::Turbo => {
                        self.ensure_model_ready(model, ctx)
                    }
                    _ => self.endpoint = LocalEndpoint::Ollama,
                }
                ctx.notify();
            }
            LocalAiChatAction::ToggleTurbo => {
                self.endpoint_user_chosen = true;
                self.cloud_active = false;
                self.error = None;
                if self.endpoint == LocalEndpoint::Turbo {
                    // Turn Turbo off -> back to the local ollama endpoint.
                    self.endpoint = LocalEndpoint::Ollama;
                } else {
                    // Turn Turbo on. refresh_models re-probes /health, so a stale
                    // turbo_available can't pin us to a dead endpoint for long.
                    self.endpoint = LocalEndpoint::Turbo;
                    // Point Turbo at the model that is actually SELECTED. Without
                    // this, switching endpoints just re-aimed the requests at
                    // whatever llama-server already had open, so the picker and
                    // the server disagreed about which model was answering.
                    if let Some(model) = self.current_model() {
                        self.ensure_model_ready(model, ctx);
                    } else if !self.turbo_available {
                        self.error = Some(
                            "Turbo (genesi-ai-turbo) isn't up yet — start it and it'll be used \
                             automatically once /health responds."
                                .to_string(),
                        );
                    }
                }
                self.model_picker_open = false;
                self.refresh_models(ctx);
                ctx.notify();
            }
            LocalAiChatAction::KeepEdits => {
                let paths = self
                    .pending_edits
                    .iter()
                    .map(|edit| edit.path.clone())
                    .collect::<Vec<_>>();
                for path in paths {
                    self.update_open_editor_diff(&path, None, ctx);
                }
                self.pending_edits.clear();
                self.review_expanded = false;
                self.selected_review_path = None;
                self.persist_mempalace(ctx, false);
                ctx.notify();
            }
            LocalAiChatAction::UndoEdits => {
                // Restore each captured original (or delete a file the agent
                // created), inside the project root only.
                let edits = std::mem::take(&mut self.pending_edits);
                if let Some(root) = self.agent_root.clone() {
                    for edit in edits {
                        let Some(path) = local_agent::resolve_in_project(&root, &edit.path) else {
                            continue;
                        };
                        match edit.original {
                            Some(content) => {
                                let _ = std::fs::write(&path, content);
                            }
                            None => {
                                let _ = std::fs::remove_file(&path);
                            }
                        }
                    }
                }
                self.pending_edits.clear();
                self.review_expanded = false;
                self.selected_review_path = None;
                self.persist_mempalace(ctx, false);
                ctx.notify();
            }
            LocalAiChatAction::OpenDiff(index) => {
                if let Some(path) = self
                    .messages
                    .get(*index)
                    .and_then(|entry| entry.tool_title.clone())
                {
                    self.review_expanded = true;
                    self.selected_review_path = Some(path);
                }
                ctx.emit(LocalAiChatEvent::OpenDiff);
                ctx.notify();
            }
            LocalAiChatAction::ToggleReviewExpanded => {
                self.review_expanded = true;
                if self.selected_review_path.is_none() {
                    self.selected_review_path =
                        self.pending_edits.first().map(|edit| edit.path.clone());
                }
                ctx.emit(LocalAiChatEvent::OpenDiff);
                ctx.notify();
            }
            LocalAiChatAction::SelectReviewFile(path) => {
                self.review_expanded = true;
                self.selected_review_path = Some(path.clone());
                ctx.emit(LocalAiChatEvent::OpenDiff);
                ctx.notify();
            }
            LocalAiChatAction::ApproveTool => {
                if let Some(tool) = self.pending_tool.take() {
                    self.start_tool(tool, ctx);
                }
                ctx.notify();
            }
            LocalAiChatAction::DenyTool => {
                if let Some(tool) = self.pending_tool.take() {
                    let name = tool.name();
                    let denied = ChatEntry {
                        role: match tool {
                            AgentTool::RunCommand { .. } => ChatRole::Command,
                            _ => ChatRole::Tool,
                        },
                        text: "(denied by user)".to_string(),
                        context_label: None,
                        tool_title: Some(tool.summary()),
                        command: match &tool {
                            AgentTool::RunCommand { command } => Some(command.clone()),
                            _ => None,
                        },
                        collapsed: false,
                        status: StepStatus::Denied,
                        diff_stat: None,
                        diff_preview: None,
                    };
                    self.messages.push(denied);
                    self.agent_messages.push(ChatMessage::user(format!(
                        "TOOL RESULT ({name}):\nThe user denied this action."
                    )));
                    self.agent_step += 1;
                    self.persist_mempalace(ctx, false);
                    self.run_agent_step(ctx);
                }
                ctx.notify();
            }
            LocalAiChatAction::Stop => {
                self.stop_turn(ctx);
            }
            LocalAiChatAction::ToggleChatPicker => {
                self.chat_picker_open = !self.chat_picker_open;
                ctx.notify();
            }
            LocalAiChatAction::PickChat(id) => {
                let id = id.clone();
                self.chat_picker_open = false;
                self.open_chat(&id, ctx);
            }
            LocalAiChatAction::DeleteChat(id) => {
                let id = id.clone();
                self.delete_chat(&id, ctx);
            }
            LocalAiChatAction::ToggleStepGroup(start) => {
                if !self.expanded_step_groups.remove(start) {
                    self.expanded_step_groups.insert(*start);
                }
                ctx.notify();
            }
            LocalAiChatAction::ToggleCollapse(index) => {
                let expanding = self
                    .messages
                    .get(*index)
                    .is_some_and(|entry| entry.collapsed);
                if expanding {
                    self.collapse_other_diffs(*index);
                }
                if let Some(entry) = self.messages.get_mut(*index) {
                    entry.collapsed = !entry.collapsed;
                }
                ctx.notify();
            }
            LocalAiChatAction::NewChat => {
                self.start_new_chat(ctx);
            }
            LocalAiChatAction::SubmitPromptInput => {
                self.input.update(ctx, |input, ctx| input.submit(ctx));
            }
            LocalAiChatAction::ToggleSoundscape => {
                self.soundscape_enabled = !self.soundscape_enabled;
                ctx.notify();
            }
            LocalAiChatAction::CycleSoundscape => {
                self.soundscape_index = (self.soundscape_index + 1) % 4;
                self.soundscape_enabled = true;
                ctx.notify();
            }
            LocalAiChatAction::ToggleKeyboardAsmr => {
                self.keyboard_asmr_enabled = !self.keyboard_asmr_enabled;
                ctx.notify();
            }
            LocalAiChatAction::CycleKeyboardSwitch => {
                self.keyboard_switch_index = (self.keyboard_switch_index + 1) % 3;
                self.keyboard_asmr_enabled = true;
                ctx.notify();
            }
        }
    }
}

impl LocalAiChatView {
    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or_default()
    }

    fn new_chat_id() -> String {
        format!("chat-{}", Self::now_ms())
    }

    fn chat_title_from_entries(entries: &[ChatEntry]) -> String {
        entries
            .iter()
            .find(|entry| entry.role == ChatRole::User && !entry.text.trim().is_empty())
            .map(|entry| truncate_middle(entry.text.trim(), 28))
            .unwrap_or_else(|| "New Chat".to_string())
    }

    fn prune_chats(&mut self) {
        let active_chat_id = self.active_chat_id.clone();
        self.chats.retain(|chat| {
            if chat.id == active_chat_id {
                return true;
            }
            !chat.messages.is_empty()
        });
        self.chats
            .sort_by(|left, right| right.updated_at_ms.cmp(&left.updated_at_ms));
    }

    fn snapshot_current_chat(&self) -> PersistedLocalChat {
        PersistedLocalChat {
            id: self.active_chat_id.clone(),
            messages: self.messages.clone(),
            agent_messages: self.agent_messages.clone(),
            updated_at_ms: Self::now_ms(),
        }
    }

    fn persist_mempalace(&mut self, ctx: &mut ViewContext<Self>, emit_state_changed: bool) {
        if self.active_chat_id.is_empty() {
            self.active_chat_id = Self::new_chat_id();
        }

        let snapshot = self.snapshot_current_chat();
        if let Some(existing) = self.chats.iter_mut().find(|chat| chat.id == snapshot.id) {
            *existing = snapshot;
        } else {
            self.chats.push(snapshot);
        }
        self.prune_chats();

        let state = MempalaceState {
            active_chat_id: self.active_chat_id.clone(),
            chats: self.chats.clone(),
            chat_mode: Some(self.chat_mode()),
        };
        match serde_json::to_string(&state) {
            Ok(json) => {
                if let Err(err) = ctx
                    .secure_storage()
                    .write_value_with_owner_only_fallback(MEMPALACE_STORAGE_KEY, &json)
                {
                    log::warn!("Failed to persist Genesi chat mempalace: {err:#}");
                }
            }
            Err(err) => {
                log::warn!("Failed to serialize Genesi chat mempalace: {err:#}");
            }
        }

        if emit_state_changed {
            ctx.emit(LocalAiChatEvent::StateChanged);
        }
    }

    fn restore_chat(&mut self, chat: &PersistedLocalChat) {
        self.messages = chat.messages.clone();
        self.agent_messages = chat.agent_messages.clone();
        self.error = None;
        self.pending_tool = None;
        self.pending_edits.clear();
        self.review_expanded = false;
        self.selected_review_path = None;
        self.agent_root = None;
        self.agent_model.clear();
        self.agent_step = 0;
        self.agent_step_buffer.clear();
        self.agent_tool_summary.clear();
        self.in_flight = false;
        self.current_turn += 1;
    }

    fn load_mempalace(&mut self, ctx: &mut ViewContext<Self>) {
        match ctx.secure_storage().read_value(MEMPALACE_STORAGE_KEY) {
            Ok(json) => match serde_json::from_str::<MempalaceState>(&json) {
                Ok(state) => {
                    self.active_chat_id = state.active_chat_id;
                    self.chats = state.chats;
                    if let Some(mode) = state.chat_mode {
                        self.set_chat_mode(mode);
                    }
                    self.prune_chats();
                    if self.active_chat_id.is_empty() {
                        self.active_chat_id = self
                            .chats
                            .first()
                            .map(|chat| chat.id.clone())
                            .unwrap_or_else(Self::new_chat_id);
                    }
                    if let Some(chat) = self
                        .chats
                        .iter()
                        .find(|chat| chat.id == self.active_chat_id)
                        .cloned()
                    {
                        self.restore_chat(&chat);
                    } else {
                        self.messages.clear();
                        self.agent_messages.clear();
                        self.persist_mempalace(ctx, false);
                    }
                }
                Err(err) => {
                    log::warn!("Failed to deserialize Genesi chat mempalace: {err:#}");
                    self.active_chat_id = Self::new_chat_id();
                    self.persist_mempalace(ctx, false);
                }
            },
            Err(err) => {
                if !matches!(err, warpui_extras::secure_storage::Error::NotFound) {
                    log::warn!("Failed to read Genesi chat mempalace: {err:#}");
                }
                self.active_chat_id = Self::new_chat_id();
                self.persist_mempalace(ctx, false);
            }
        }
    }

    pub fn chat_summaries(&self) -> Vec<LocalChatSummary> {
        let mut chats = self.chats.clone();
        if !self.active_chat_id.is_empty() {
            let snapshot = self.snapshot_current_chat();
            if let Some(existing) = chats.iter_mut().find(|chat| chat.id == snapshot.id) {
                *existing = snapshot;
            } else {
                chats.push(snapshot);
            }
        }
        chats.sort_by(|left, right| right.updated_at_ms.cmp(&left.updated_at_ms));
        chats
            .into_iter()
            .map(|chat| LocalChatSummary {
                id: chat.id.clone(),
                title: Self::chat_title_from_entries(&chat.messages),
                is_active: chat.id == self.active_chat_id,
            })
            .collect()
    }

    pub fn open_chat(&mut self, chat_id: &str, ctx: &mut ViewContext<Self>) {
        if chat_id == self.active_chat_id {
            return;
        }

        self.stop_turn(ctx);
        self.persist_mempalace(ctx, false);

        if let Some(chat) = self.chats.iter().find(|chat| chat.id == chat_id).cloned() {
            self.active_chat_id = chat.id.clone();
            self.restore_chat(&chat);
            self.scroll_to_bottom();
            ctx.emit(LocalAiChatEvent::StateChanged);
            ctx.notify();
        }
    }

    pub fn delete_chat(&mut self, chat_id: &str, ctx: &mut ViewContext<Self>) {
        self.stop_turn(ctx);

        let was_active = chat_id == self.active_chat_id;
        if !was_active {
            self.persist_mempalace(ctx, false);
        }

        self.chats.retain(|chat| chat.id != chat_id);

        if was_active {
            if let Some(next_chat) = self
                .chats
                .iter()
                .max_by_key(|chat| chat.updated_at_ms)
                .cloned()
            {
                self.active_chat_id = next_chat.id.clone();
                self.restore_chat(&next_chat);
                self.scroll_to_bottom();
            } else {
                self.active_chat_id = Self::new_chat_id();
                self.reset_current_chat();
            }
        }

        self.persist_mempalace(ctx, true);
        ctx.notify();
    }
}

impl View for LocalAiChatView {
    fn ui_name() -> &'static str {
        "LocalAiChatView"
    }

    fn on_focus(&mut self, focus_ctx: &FocusContext, ctx: &mut ViewContext<Self>) {
        if focus_ctx.is_self_focused() {
            ctx.focus(&self.input);
        }
    }

    fn render(&self, app: &AppContext) -> Box<dyn Element> {
        let appearance = Appearance::as_ref(app);
        let theme = appearance.theme();

        let mut root = Flex::column().with_main_axis_size(MainAxisSize::Max);
        if self.vibe_mode {
            // Center the chat column on the X axis (Claude-style) instead of
            // letting it hug the left edge of the wide vibe main area. The
            // transcript and the compose box below are both width-capped at 760
            // so this reads as a centered conversation column.
            root = root.with_cross_axis_alignment(CrossAxisAlignment::Center);
        }

        let header = if self.vibe_mode {
            Container::new(
                ConstrainedBox::new(self.render_header(appearance))
                    .with_width(VIBE_COLUMN_WIDTH)
                    .finish(),
            )
            .with_horizontal_padding(PANEL_PADDING)
            .with_padding_top(PANEL_PADDING + 4.)
            .with_padding_bottom(4.)
            .finish()
        } else {
            Container::new(self.render_header(appearance))
                .with_uniform_padding(PANEL_PADDING)
                .with_border(Border::bottom(1.).with_border_fill(theme.outline()))
                .finish()
        };
        root.add_child(header);

        let transcript: Box<dyn Element> = if self.vibe_mode {
            // Cap the transcript width to match the 760px compose box so the
            // centered column reads like a chat instead of stretching full-width.
            ConstrainedBox::new(self.render_transcript(appearance))
                .with_max_width(VIBE_COLUMN_WIDTH)
                .finish()
        } else {
            self.render_transcript(appearance)
        };
        root.add_child(Expanded::new(1., transcript).finish());

        if let Some(error) = &self.error {
            root.add_child(
                Container::new(self.label_text(
                    appearance,
                    error.clone(),
                    CHIP_FONT_SIZE,
                    theme.ui_error_color().into(),
                    true,
                ))
                .with_horizontal_padding(PANEL_PADDING)
                .with_padding_bottom(4.)
                .finish(),
            );
        }

        // A pending command waits for the user's Allow/Deny above the input.
        if let Some(tool) = &self.pending_tool {
            root.add_child(self.render_approval(appearance, tool));
        }

        // Review bar: a summary of the agent's file changes with Undo all / Keep,
        // sitting just above the compose box (the reference's "N files need review").
        if let Some(bar) = self.render_review_bar(appearance) {
            root.add_child(bar);
        }

        // AI model picker popup — opens ABOVE the compose box (IDE-style): the
        // user clicks the selector that lives inside the box and the list pops
        // up over it.
        if let Some(picker) = self.render_model_picker(appearance) {
            root.add_child(picker);
        }
        if let Some(picker) = self.render_mode_picker(appearance) {
            root.add_child(picker);
        }
        if let Some(picker) = self.render_chat_picker(appearance) {
            root.add_child(picker);
        }

        // The compose box: a single full-width, bordered, rounded container that
        // holds the prompt input on top and the AI controls (selector + Turbo +
        // agent toggles) along its bottom edge — like a real IDE assistant, with
        // everything in one surface instead of loose chips above a bare input.
        // The prompt field spans the full width on its own line, and every
        // control — including send — sits on the row beneath it, so the send
        // affordance lands in the box's bottom-right corner like the reference.
        let mut compose_inner =
            Flex::column().with_cross_axis_alignment(CrossAxisAlignment::Stretch);
        // Reference chips sit above the prompt, where the user can see what the
        // next message carries before sending it.
        if let Some(attachments) = self.render_attachment_row(appearance) {
            compose_inner.add_child(attachments);
        }
        let compose_inner = compose_inner
            .with_child(
                Container::new(ChildView::new(&self.input).finish())
                    .with_horizontal_padding(4.)
                    .with_vertical_padding(2.)
                    .finish(),
            )
            .with_child(
                Container::new(self.render_control_strip(appearance))
                    .with_padding_top(6.)
                    .finish(),
            )
            .finish();
        let compose_inner = if self.vibe_mode {
            Flex::column()
                .with_cross_axis_alignment(CrossAxisAlignment::Stretch)
                .with_child(compose_inner)
                .with_child(
                    Container::new(self.render_vibe_lab_strip(appearance))
                        .with_padding_top(6.)
                        .finish(),
                )
                .finish()
        } else {
            compose_inner
        };
        // No border: the card is separated from the panel by being LIGHTER, not by
        // an outline. genesi_card_surface sat within a point or two of the panel
        // fill, so the box was invisible against it.
        // The compose box doubles as a drop target so a file dragged out of the
        // file tree lands in the conversation as a reference.
        let compose_box = DropTarget::new(
            Container::new(compose_inner)
                .with_horizontal_padding(11.)
                .with_vertical_padding(10.)
                .with_corner_radius(CornerRadius::with_all(Radius::Pixels(14.)))
                .with_background_color(genesi_compose_surface())
                .finish(),
            LocalAiDropTargetData {
                panel: self.weak_handle.clone(),
            },
        )
        .finish();
        let compose_container = if self.vibe_mode {
            Container::new(
                ConstrainedBox::new(compose_box)
                    .with_width(VIBE_COLUMN_WIDTH)
                    .finish(),
            )
            .with_horizontal_padding(PANEL_PADDING)
            .with_padding_top(4.)
            .with_padding_bottom(PANEL_PADDING + 8.)
            .finish()
        } else {
            Container::new(compose_box)
                .with_horizontal_padding(PANEL_PADDING)
                .with_padding_top(4.)
                .with_padding_bottom(PANEL_PADDING)
                .finish()
        };
        root.add_child(compose_container);

        // Vibe mode paints NO surface of its own — it sits on the normal app
        // background like every other view. A solid tinted fill here turned the
        // whole mode into one flat colour block.
        root.finish()
    }
}
