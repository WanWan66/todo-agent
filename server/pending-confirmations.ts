import { randomUUID } from "node:crypto";
import type { ChatMessage } from "../shared/chat.ts";
import type { NewTaskInput } from "./task-store.ts";

export type { NewSubtaskInput, NewTaskInput } from "./task-store.ts";

export type ConfirmationScope = { channel: "chat"; conversationId: string } | { channel: "rest" };

export type PendingConfirmationAction =
  | {
      kind: "duplicate-create";
      channel: "chat";
      conversationId: string;
      tasks: NewTaskInput[];
      duplicateTaskIds: string[];
      duplicateTitles: string[];
    }
  | ({
      kind: "cascade-delete";
      rootTaskIds: string[];
      affectedTaskIds: string[];
    } & ConfirmationScope);

export type PendingConfirmation = PendingConfirmationAction & {
  confirmationId: string;
  createdAt: number;
  expiresAt: number;
};

type ConfirmationFailure = { ok: false; reason: "NOT_FOUND" | "EXPIRED" | "FORBIDDEN" };
export type ConsumeConfirmationResult =
  | { ok: true; confirmation: PendingConfirmation }
  | ConfirmationFailure;
export type CancelConfirmationResult = { ok: true } | ConfirmationFailure;

export interface PendingConfirmationStore {
  create(action: PendingConfirmationAction): { confirmationId: string; expiresAt: number };
  getByConversationId(conversationId: string): PendingConfirmation | undefined;
  getById(confirmationId: string): PendingConfirmation | undefined;
  consume(confirmationId: string, expected: ConfirmationScope): ConsumeConfirmationResult;
  cancel(confirmationId: string, expected: ConfirmationScope): CancelConfirmationResult;
}

export function createPendingConfirmationStore({
  ttlMs = 600_000,
  now = Date.now,
}: { ttlMs?: number; now?: () => number } = {}): PendingConfirmationStore {
  const confirmations = new Map<string, PendingConfirmation>();
  const chatIds = new Map<string, string>();

  function getById(confirmationId: string) {
    const confirmation = confirmations.get(confirmationId);
    // 过期记录留到首次确认或取消，以便区分已过期和不存在。
    if (confirmation && now() < confirmation.expiresAt) return structuredClone(confirmation);
    return undefined;
  }

  function consume(confirmationId: string, expected: ConfirmationScope): ConsumeConfirmationResult {
    const confirmation = confirmations.get(confirmationId);
    if (!confirmation) return { ok: false, reason: "NOT_FOUND" };
    if (
      expected?.channel !== confirmation.channel ||
      (expected.channel === "chat" &&
        confirmation.channel === "chat" &&
        expected.conversationId !== confirmation.conversationId)
    ) {
      return { ok: false, reason: "FORBIDDEN" };
    }

    const expired = now() >= confirmation.expiresAt;
    confirmations.delete(confirmationId);
    if (
      confirmation.channel === "chat" &&
      chatIds.get(confirmation.conversationId) === confirmationId
    ) {
      chatIds.delete(confirmation.conversationId);
    }
    if (expired) return { ok: false, reason: "EXPIRED" };
    return { ok: true, confirmation };
  }

  return {
    create(action) {
      const createdAt = now();
      if (action.channel === "chat") {
        const previousId = chatIds.get(action.conversationId);
        const previous = previousId === undefined ? undefined : confirmations.get(previousId);
        if (previous && createdAt < previous.expiresAt) {
          throw new Error("PENDING_CONFIRMATION_EXISTS");
        }
      }

      const confirmationId = randomUUID();
      const expiresAt = createdAt + ttlMs;
      confirmations.set(confirmationId, {
        ...structuredClone(action),
        confirmationId,
        createdAt,
        expiresAt,
      });
      if (action.channel === "chat") chatIds.set(action.conversationId, confirmationId);
      return { confirmationId, expiresAt };
    },

    getByConversationId(conversationId) {
      const confirmationId = chatIds.get(conversationId);
      return confirmationId === undefined ? undefined : getById(confirmationId);
    },

    getById,
    consume,

    cancel(confirmationId, expected) {
      const result = consume(confirmationId, expected);
      return result.ok ? { ok: true } : result;
    },
  };
}

export type ConfirmationIntent = "confirm" | "cancel" | "unclear";
export type StaleConfirmationIntent = "confirm" | "cancel" | "none";

const staleConfirmPhrases = new Set([
  "确认",
  "确定",
  "确认删除",
  "确认创建",
  "确定删除",
  "确定创建",
]);
const staleCancelPhrases = new Set(["取消", "取消删除", "取消创建"]);
const confirmPhrases = new Set([
  ...staleConfirmPhrases,
  "是的",
  "对",
  "好",
  "好的",
  "可以",
  "同意",
]);
const cancelPhrases = new Set([
  ...staleCancelPhrases,
  "不用了",
  "不用",
  "不要",
  "算了",
  "不删了",
  "不建了",
]);

function normalizeConfirmationMessage(message: string) {
  return message.trim().replace(/[。.!！?？,，~～]+$/u, "");
}

export function parseConfirmationIntent(message: string): ConfirmationIntent {
  const normalized = normalizeConfirmationMessage(message);
  if (confirmPhrases.has(normalized)) return "confirm";
  if (cancelPhrases.has(normalized)) return "cancel";
  return "unclear";
}

export function parseStaleConfirmationIntent(message: string): StaleConfirmationIntent {
  const normalized = normalizeConfirmationMessage(message);
  if (staleConfirmPhrases.has(normalized)) return "confirm";
  if (staleCancelPhrases.has(normalized)) return "cancel";
  return "none";
}

export interface StaleContext {
  hasRecentUnresolvedConfirmationPrompt: boolean;
}

export function hasRecentUnresolvedConfirmationPrompt(messages: ChatMessage[]): boolean {
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message.role !== "assistant") continue;
    if (message.metadata?.kind === "confirmation_resolved") return false;
    if (message.metadata?.kind === "pending_confirmation_prompt") return true;
  }
  return false;
}

export function shouldBlockStaleAcknowledgement(message: string, context: StaleContext): boolean {
  return (
    context.hasRecentUnresolvedConfirmationPrompt && parseConfirmationIntent(message) !== "unclear"
  );
}

export type WriteOperationKind = "create" | "update" | "delete";

export interface CurrentTurnWriteIntent {
  operation: WriteOperationKind;
  targetText: string;
  sourceMessage: string;
  allowedTools:
    | ["list_tasks", "create_tasks"]
    | ["list_tasks", "update_tasks"]
    | ["list_tasks", "delete_tasks"];
}

export type TurnToolPolicy =
  | { mode: "read-only"; allowedTools: ["list_tasks"]; requiresClarification?: boolean }
  | {
      mode: "scoped-write";
      allowedTools: CurrentTurnWriteIntent["allowedTools"];
      currentTurnIntent: CurrentTurnWriteIntent;
    }
  | {
      mode: "unrestricted";
      allowedTools: ["list_tasks", "create_tasks", "update_tasks", "delete_tasks"];
    };

const writePatterns = [
  {
    operation: "create",
    pattern: /^(?:再\s*)?(?:添加|创建|新建|新增|记一下|记一条|提醒我)\s*(.+)$/u,
  },
  {
    operation: "update",
    field: "done",
    pattern: /^(?:把|将)\s*(.+?)\s*(?:标记|设)为(已完成|未完成)$/u,
  },
  { operation: "update", field: "title", pattern: /^(?:把|将)\s*(.+?)\s*改名为\s*(.+)$/u },
  { operation: "delete", pattern: /^(?:删除|删掉|移除)\s*(.+)$/u },
  { operation: "delete", pattern: /^(?:把|将)\s*(.+?)\s*(?:这条\s*)?(?:删除|删掉|移除)(?:吧)?$/u },
  { operation: "delete", pattern: /^(.+?)\s*这条\s*(?:删除|删掉|移除)(?:吧)?$/u },
] as const;

const writeTools: Record<WriteOperationKind, CurrentTurnWriteIntent["allowedTools"][1]> = {
  create: "create_tasks",
  update: "update_tasks",
  delete: "delete_tasks",
};

export function explicitTaskTarget(text: string): string | undefined {
  const title = text.trim();
  if (!title || /[\n\r]/u.test(title)) return undefined;
  const quoted = /^(?:「([^」]+)」|『([^』]+)』|“([^”]+)”|"([^"]+)")$/u.exec(title);
  if (quoted) {
    const [, corner, doubleCorner, curly, straight] = quoted;
    const inner = (corner ?? doubleCorner ?? curly ?? straight).trim();
    return inner && inner.length <= 200 ? inner : undefined;
  }

  // 只从明确命名的单一目标提取权限。指代、疑问和组合指令继续走只读。
  const reference =
    /^(?:[这那哪][个项些条一]|[上下][个条一次]|刚[才刚]|之前|前面|原来|所有|全部|其余|剩[下余]|其他|其中|最后一|第[一二三四五六七八九十\d])|^(?:它们?|任务|待办|清单)$/u;
  const ambiguous =
    /[，,、；;。.!！?？「」『』“”"]|.[和与或].|(?:然后|并且|顺便|以及|或者|还是|按刚才|先不要|如果|假如|顺手|同时|[并再]\s*(?:把|将|设为|标记|改名|添加|创建|删除|删掉|更新|修改))|(?:吗|如何|怎么做|怎么办|的任务|的待办)$/u;
  if (title.length > 200 || reference.test(title) || ambiguous.test(title)) return undefined;
  return title;
}

type ExplicitTaskWrite =
  | { operation: "create"; targetText: string }
  | { operation: "delete"; targetText: string }
  | { operation: "update"; targetText: string; changes: { done: boolean } | { title: string } };

export function parseExplicitTaskWrite(message: string): ExplicitTaskWrite | undefined {
  const command = message
    .trim()
    .replace(/[。.!！]+$/u, "")
    .replace(/^(?:现在\s*)?(?:请\s*)?(?:帮我\s*)?(?:重新\s*)?/u, "");
  for (const rule of writePatterns) {
    const match = rule.pattern.exec(command);
    if (!match) continue;
    const targetText = explicitTaskTarget(match[1]);
    if (!targetText) continue;
    if (rule.operation !== "update") return { operation: rule.operation, targetText };
    if (rule.field === "done") {
      return { operation: "update", targetText, changes: { done: match[2] === "已完成" } };
    }
    const title = explicitTaskTarget(match[2]);
    if (title) return { operation: "update", targetText, changes: { title } };
  }
  return undefined;
}

export function determineTurnToolPolicy(message: string, context: StaleContext): TurnToolPolicy {
  if (!context.hasRecentUnresolvedConfirmationPrompt) {
    return {
      mode: "unrestricted",
      allowedTools: ["list_tasks", "create_tasks", "update_tasks", "delete_tasks"],
    };
  }
  const write = parseExplicitTaskWrite(message);
  if (write) {
    const { operation, targetText } = write;
    const allowedTools: CurrentTurnWriteIntent["allowedTools"] = [
      "list_tasks",
      writeTools[operation],
    ];
    return {
      mode: "scoped-write",
      allowedTools,
      currentTurnIntent: { operation, targetText, sourceMessage: message, allowedTools },
    };
  }

  const isTaskQuery =
    /(?:哪些|什么|多少).*(?:任务|待办)|(?:任务|待办).*(?:哪些|什么|多少)|(?:查看|看看|显示|列出).*(?:任务|待办|清单)/u.test(
      message,
    );
  return { mode: "read-only", allowedTools: ["list_tasks"], requiresClarification: !isTaskQuery };
}
