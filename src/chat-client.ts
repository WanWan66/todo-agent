import * as z from "zod/mini";
import { chatMessageMetadataSchema, type ChatMessage } from "../shared/chat.ts";
import { dateInTimeZone } from "../shared/task-display.ts";

const chatResponseSchema = z.object({
  reply: z.string(),
  metadata: z.optional(chatMessageMetadataSchema),
});
const taskSchema = z.object({
  id: z.string(),
  title: z.string(),
  done: z.boolean(),
  dueDate: z.optional(z.string()),
  priority: z.optional(z.union([z.enum(["low", "medium", "high"]), z.number()])),
  effort: z.optional(z.enum(["small", "medium", "large"])),
  parentId: z.optional(z.string()),
});
const tasksSchema = z.array(taskSchema);
const confirmationSchema = z.object({
  error: z.literal("CONFIRMATION_REQUIRED"),
  confirmationId: z.string(),
  affectedTasks: tasksSchema,
  message: z.string(),
});
const errorSchema = z.object({ error: z.string(), message: z.optional(z.string()) });

export type Task = z.infer<typeof taskSchema>;
export type DeleteConfirmation = z.infer<typeof confirmationSchema>;
export interface TaskSnapshot {
  tasks: Task[];
  today: string;
}

function apiError(body: unknown) {
  const parsed = errorSchema.safeParse(body);
  return new Error(parsed.success ? parsed.data.message || parsed.data.error : "服务暂时不可用。");
}

async function readJson(response: Response): Promise<unknown> {
  return response.json().catch(() => undefined);
}

async function expectSuccess(response: Response) {
  if (!response.ok) throw apiError(await readJson(response));
}

const collection = "/api/tasks";
const taskPath = (id: string) => collection + "/" + encodeURIComponent(id);
const confirmationPath = (id: string) => collection + "/confirmations/" + encodeURIComponent(id);

export async function requestReply(messages: ChatMessage[], conversationId: string) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ messages, conversationId }),
  });
  const body = await readJson(response);
  if (!response.ok) throw apiError(body);
  const result = chatResponseSchema.safeParse(body);
  if (!result.success) throw new Error("收到的对话响应格式无效。");
  return result.data;
}

export async function fetchTasks(): Promise<TaskSnapshot> {
  const response = await fetch(collection, { cache: "no-store" });
  const body = await readJson(response);
  if (!response.ok) throw apiError(body);
  const result = tasksSchema.safeParse(body);
  if (!result.success) throw new Error("收到的任务列表格式无效。");
  const date = response.headers.get("X-Task-Today");
  return { tasks: result.data, today: date ?? dateInTimeZone(new Date()) };
}

export async function updateTaskDone(id: string, done: boolean) {
  await expectSuccess(
    await fetch(taskPath(id), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ done }),
    }),
  );
}

export async function deleteTask(id: string): Promise<DeleteConfirmation | undefined> {
  const response = await fetch(taskPath(id), { method: "DELETE" });
  if (response.status === 204) return undefined;
  const body = await readJson(response);
  if (response.status === 409) {
    const parsed = confirmationSchema.safeParse(body);
    if (parsed.success) return parsed.data;
  }
  throw apiError(body);
}

export async function confirmTaskDeletion(confirmationId: string) {
  await expectSuccess(await fetch(confirmationPath(confirmationId), { method: "POST" }));
}

export async function cancelTaskDeletion(confirmationId: string) {
  await expectSuccess(await fetch(confirmationPath(confirmationId), { method: "DELETE" }));
}
