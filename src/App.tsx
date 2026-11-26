import { useCallback, useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import type { ChatMessage } from "../shared/chat.ts";
import { dateInTimeZone, effortLabels, normalizePriority } from "../shared/task-display.ts";
import {
  ApiError,
  cancelTaskDeletion,
  confirmTaskDeletion,
  deleteTask,
  fetchTasks,
  requestReply,
  updateTaskDone,
  type DeleteConfirmation,
  type Task,
  type TaskSnapshot,
} from "./chat-client.ts";
import { ChatHistory } from "./ChatHistory.tsx";

function isSendShortcut(event: KeyboardEvent<HTMLTextAreaElement>) {
  // 中文输入法会用 Enter 确认选词，部分浏览器此时只报告键码 229。
  const composing = event.nativeEvent.isComposing || event.nativeEvent.keyCode === 229;
  return event.key === "Enter" && !event.shiftKey && !composing;
}

interface BoardProps {
  snapshot: TaskSnapshot;
  open: boolean;
  loading: boolean;
  busy: boolean;
  loadError: string;
  actionError: string;
  onToggle: () => void;
  onRefresh: () => void;
  onCompletion: (task: Task, done: boolean) => void;
  onDelete: (task: Task) => void;
}

function groupTasks(tasks: Task[]) {
  const ids = new Set(tasks.map((task) => task.id));
  const ordered = tasks.slice().sort((left, right) => {
    const rank = normalizePriority(right.priority).rank - normalizePriority(left.priority).rank;
    return rank || (left.dueDate ?? "9999-99-99").localeCompare(right.dueDate ?? "9999-99-99");
  });
  const byParent = new Map<string, Task[]>();
  const roots = [];
  for (const task of ordered) {
    if (task.parentId && ids.has(task.parentId)) {
      const children = byParent.get(task.parentId) ?? [];
      children.push(task);
      byParent.set(task.parentId, children);
    } else {
      roots.push(task);
    }
  }
  return { roots, byParent };
}

interface TaskRowProps {
  task: Task;
  childrenTasks?: Task[];
  today: string;
  busy: boolean;
  onCompletion: BoardProps["onCompletion"];
  onDelete: BoardProps["onDelete"];
}

function TaskRow({ task, childrenTasks, today, busy, onCompletion, onDelete }: TaskRowProps) {
  const priority = normalizePriority(task.priority);
  const overdue = !task.done && task.dueDate !== undefined && task.dueDate < today;
  const dueToday = !task.done && task.dueDate === today;
  return (
    <div className={task.done ? "task-row task-done" : "task-row"}>
      <input
        type="checkbox"
        checked={task.done}
        disabled={busy}
        aria-label={(task.done ? "恢复为待处理 " : "标记完成 ") + task.title}
        onChange={(event) => onCompletion(task, event.currentTarget.checked)}
      />
      <div className="task-copy">
        <span className="task-title">{task.title}</span>
        {(priority.label || task.effort || task.dueDate || childrenTasks?.length) && (
          <div className="task-details">
            {priority.label && (
              <span className={"task-tag priority-" + priority.rank}>优先级 {priority.label}</span>
            )}
            {task.effort && <span className="task-tag">工作量 {effortLabels[task.effort]}</span>}
            {task.dueDate && (
              <time
                className={
                  overdue ? "task-tag overdue" : dueToday ? "task-tag due-today" : "task-tag"
                }
                dateTime={task.dueDate}
              >
                {overdue ? "已逾期 " : dueToday ? "今日到期 " : "到期 "}
                {task.dueDate}
              </time>
            )}
            {!!childrenTasks?.length && (
              <span className="task-tag">
                子任务 {childrenTasks.filter((child) => child.done).length}/{childrenTasks.length}
              </span>
            )}
          </div>
        )}
      </div>
      <button
        type="button"
        className="task-delete"
        aria-label={"删除 " + task.title}
        disabled={busy}
        onClick={() => onDelete(task)}
      >
        删除
      </button>
    </div>
  );
}

function TaskBoard({
  snapshot,
  open,
  loading,
  busy,
  loadError,
  actionError,
  onToggle,
  onRefresh,
  onCompletion,
  onDelete,
}: BoardProps) {
  const { tasks, today } = snapshot;
  const { roots, byParent } = useMemo(() => groupTasks(tasks), [tasks]);
  const groups = [
    { title: "待处理", entries: roots.filter((task) => !task.done) },
    { title: "已完成", entries: roots.filter((task) => task.done) },
  ];
  const rowProps = { busy, today, onCompletion, onDelete };
  return (
    <aside className="task-board" aria-labelledby="task-board-title">
      <div className="board-header">
        <div>
          <h2 id="task-board-title">待办看板</h2>
          {open && <p>{tasks.filter((task) => !task.done).length} 条待处理</p>}
        </div>
        <button
          type="button"
          className="board-toggle"
          aria-expanded={open}
          aria-controls="task-board-content"
          onClick={onToggle}
        >
          {open ? "收起" : "展开"}
        </button>
      </div>
      <div className="board-content" id="task-board-content" hidden={!open} aria-busy={loading}>
        {actionError && (
          <p className="error board-error" role="alert">
            {actionError}
          </p>
        )}
        {loadError && (
          <div className="board-error" role="alert">
            <p>{loadError}</p>
            <button type="button" className="board-toggle" onClick={onRefresh}>
              重试
            </button>
          </div>
        )}
        {!tasks.length && !loadError && (
          <p className="board-empty" role="status">
            {loading ? "正在读取任务…" : "这里还没有任务。"}
          </p>
        )}
        {groups.map(
          ({ title, entries }) =>
            entries.length > 0 && (
              <section className="task-group" key={title} aria-label={title}>
                <h3>{title}</h3>
                <ul className="task-list">
                  {entries.map((task) => {
                    const children = byParent.get(task.id);
                    return (
                      <li className="task-card" key={task.id}>
                        <TaskRow task={task} childrenTasks={children} {...rowProps} />
                        {children && (
                          <ul className="task-children" aria-label={task.title + "的子任务"}>
                            {children.map((child) => (
                              <li key={child.id}>
                                <TaskRow task={child} {...rowProps} />
                              </li>
                            ))}
                          </ul>
                        )}
                      </li>
                    );
                  })}
                </ul>
              </section>
            ),
        )}
      </div>
    </aside>
  );
}

function DeleteDialog({
  confirmation,
  busy,
  error,
  onDecision,
}: {
  confirmation: DeleteConfirmation;
  busy: boolean;
  error: string;
  onDecision: (accept: boolean) => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = dialog.current;
    element?.showModal();
    return () => element?.close();
  }, []);
  return (
    <dialog
      className="deletion-dialog"
      ref={dialog}
      aria-labelledby="delete-dialog-title"
      onCancel={(event) => {
        event.preventDefault();
        if (!busy) onDecision(false);
      }}
    >
      <h2 id="delete-dialog-title">确认删除任务</h2>
      <p>{confirmation.message}</p>
      <ul>
        {confirmation.affectedTasks.map((task) => (
          <li key={task.id}>
            {task.title}
            {task.parentId && <small>子任务</small>}
          </li>
        ))}
      </ul>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <div className="dialog-actions">
        <button
          type="button"
          className="board-toggle"
          disabled={busy}
          onClick={() => onDecision(false)}
        >
          取消
        </button>
        <button
          type="button"
          className="danger-button"
          disabled={busy}
          onClick={() => onDecision(true)}
        >
          确认删除
        </button>
      </div>
    </dialog>
  );
}

export default function App() {
  const [conversationId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [snapshot, setSnapshot] = useState<TaskSnapshot>(() => ({
    tasks: [],
    today: dateInTimeZone(new Date()),
  }));
  const [boardOpen, setBoardOpen] = useState(true);
  const [boardLoading, setBoardLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [actionError, setActionError] = useState("");
  const [boardBusy, setBoardBusy] = useState(false);
  const [confirmation, setConfirmation] = useState<DeleteConfirmation>();
  const [dialogError, setDialogError] = useState("");
  const input = useRef<HTMLTextAreaElement>(null);
  const sending = useRef(false);
  const taskAction = useRef(false);
  const refreshSequence = useRef(0);

  const refreshTasks = useCallback(async () => {
    const sequence = ++refreshSequence.current;
    setBoardLoading(true);
    setLoadError("");
    try {
      const next = await fetchTasks();
      if (sequence === refreshSequence.current) setSnapshot(next);
    } catch (cause) {
      if (sequence === refreshSequence.current) {
        setLoadError(cause instanceof Error ? "读取任务失败。 " + cause.message : "读取任务失败。");
      }
    } finally {
      if (sequence === refreshSequence.current) setBoardLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshTasks();
    return () => {
      refreshSequence.current++;
    };
  }, [refreshTasks]);

  async function runTaskAction(action: () => Promise<void>) {
    if (taskAction.current) return;
    taskAction.current = true;
    setBoardBusy(true);
    setActionError("");
    try {
      await action();
    } catch (cause) {
      setActionError(cause instanceof Error ? cause.message : "操作失败，请稍后重试。");
    } finally {
      await refreshTasks();
      taskAction.current = false;
      setBoardBusy(false);
    }
  }

  async function send(text: string) {
    const content = text.trim();
    if (!content || sending.current) return;
    sending.current = true;
    const outgoing: ChatMessage[] = [...messages, { role: "user", content }];
    setMessages(outgoing);
    setDraft("");
    setPending(true);
    setError("");
    try {
      const { reply, metadata } = await requestReply(outgoing, conversationId);
      const answer: ChatMessage = metadata
        ? { role: "assistant", content: reply, metadata }
        : { role: "assistant", content: reply };
      setMessages([...outgoing, answer]);
    } catch (cause) {
      setMessages(messages);
      setDraft(text);
      setError(cause instanceof Error ? `发送失败：${cause.message}` : "发送失败，请重试。");
    } finally {
      await refreshTasks();
      sending.current = false;
      setPending(false);
      requestAnimationFrame(() => input.current?.focus());
    }
  }

  return (
    <div className="app">
      <header>
        <img src="/logo.png" alt="" />
        <strong>TokeNow</strong>
        <span>小明智能体</span>
      </header>
      <main className={boardOpen ? "workspace" : "workspace board-collapsed"}>
        <section className="chat-pane" aria-label="待办对话">
          <ChatHistory messages={messages} pending={pending} onSelectPrompt={send} />
          <div className="compose">
            {error && (
              <p className="error" role="alert">
                {error}
              </p>
            )}
            <form
              onSubmit={(event) => {
                event.preventDefault();
                void send(draft);
              }}
            >
              <textarea
                aria-label="消息"
                ref={input}
                placeholder="输入消息…"
                rows={2}
                value={draft}
                readOnly={pending}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (isSendShortcut(event)) {
                    event.preventDefault();
                    void send(draft);
                  }
                }}
              />
              <button className="send" disabled={pending || !draft.trim()} aria-label="发送消息">
                ↑
              </button>
            </form>
            <small>Enter 发送 · Shift + Enter 换行</small>
          </div>
        </section>
        <TaskBoard
          snapshot={snapshot}
          open={boardOpen}
          loading={boardLoading}
          busy={boardBusy}
          loadError={loadError}
          actionError={actionError}
          onToggle={() => setBoardOpen((open) => !open)}
          onRefresh={() => {
            void refreshTasks();
          }}
          onCompletion={(task, done) => {
            void runTaskAction(() => updateTaskDone(task.id, done));
          }}
          onDelete={(task) => {
            void runTaskAction(async () => {
              setDialogError("");
              setConfirmation(await deleteTask(task.id));
            });
          }}
        />
      </main>
      {confirmation && (
        <DeleteDialog
          confirmation={confirmation}
          busy={boardBusy}
          error={dialogError}
          onDecision={(accept) => {
            void runTaskAction(async () => {
              setDialogError("");
              try {
                const request = accept ? confirmTaskDeletion : cancelTaskDeletion;
                await request(confirmation.confirmationId);
                setConfirmation(undefined);
              } catch (cause) {
                if (cause instanceof ApiError && cause.status >= 400 && cause.status < 500) {
                  setConfirmation(undefined);
                  throw cause;
                }
                setDialogError(
                  cause instanceof Error
                    ? `${cause.message}，请重试。`
                    : "网络或服务异常，请重试。",
                );
              }
            });
          }}
        />
      )}
    </div>
  );
}
