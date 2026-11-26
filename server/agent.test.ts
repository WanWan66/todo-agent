import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { mkdtempSync, readFileSync, rmSync, rmdirSync, writeFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, beforeEach, test, type TestContext } from "node:test";
import type { ChatCompletionMessageToolCall } from "openai/resources/chat/completions";
import type { ChatMessage, ChatMessageMetadata } from "../shared/chat.ts";
import { reply, type AgentReplyOptions, type ReplyHandler } from "./agent.ts";
import { createApp } from "./app.ts";
import { model, type ChatCompletionClientLike } from "./model.ts";
import {
  createPendingConfirmationStore,
  determineTurnToolPolicy,
  hasRecentUnresolvedConfirmationPrompt,
  parseConfirmationIntent,
  parseStaleConfirmationIntent,
  type ConfirmationScope,
} from "./pending-confirmations.ts";
import { createJsonTaskStore, isValidIsoDate, normalizePriority, type Task } from "./task-store.ts";

const envKeys = ["DEEPSEEK_API_KEY", "DEEPSEEK_FALLBACK_MODEL", "APP_TIME_ZONE"] as const;
const originalEnv = new Map<(typeof envKeys)[number], string | undefined>();
before(() => {
  for (const key of envKeys) originalEnv.set(key, process.env[key]);
});
beforeEach(() => {
  for (const key of envKeys) delete process.env[key];
});
after(() => {
  for (const [key, value] of originalEnv) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
});

type ModelRequest = Parameters<ChatCompletionClientLike["chat"]["completions"]["create"]>[0];
type ModelAnswer = Awaited<
  ReturnType<ChatCompletionClientLike["chat"]["completions"]["create"]>
>["choices"][number]["message"];
type ModelStep = ModelAnswer | ((request: ModelRequest) => ModelAnswer);
type ToolResult =
  | { success: true; data: unknown; message: string }
  | { success: false; error: string; message: string; details?: unknown };

function fakeModel(...steps: ModelStep[]) {
  const requests: ModelRequest[] = [];
  const client: ChatCompletionClientLike = {
    chat: {
      completions: {
        async create(request) {
          const step = steps[requests.length];
          requests.push(request);
          assert(step !== undefined, "Fake Model received an unexpected call");
          return { choices: [{ message: typeof step === "function" ? step(request) : step }] };
        },
      },
    },
  };
  return { client, requests };
}

let nextCall = 0;
function call(name: string, args: unknown): ChatCompletionMessageToolCall {
  return {
    type: "function",
    id: "call-" + ++nextCall,
    function: { name, arguments: JSON.stringify(args) },
  };
}
const using = (...tool_calls: ChatCompletionMessageToolCall[]): ModelAnswer => ({ tool_calls });
const saying = (content: string): ModelAnswer => ({ content });
const throwing =
  (error: unknown): ModelStep =>
  () => {
    throw error;
  };

function resultFor(request: ModelRequest, tool: ChatCompletionMessageToolCall): ToolResult {
  const message = request.messages.find(
    (item) => item.role === "tool" && item.tool_call_id === tool.id,
  );
  assert(
    message?.role === "tool" && typeof message.content === "string",
    "Missing tool result " + tool.id,
  );
  return JSON.parse(message.content) as ToolResult;
}

function expectToolError(
  request: ModelRequest,
  tool: ChatCompletionMessageToolCall,
  error: string,
) {
  const result = resultFor(request, tool);
  assert.equal(result.success, false);
  if (result.success) assert.fail("Tool unexpectedly succeeded");
  assert.equal(result.error, error);
  return result;
}

function expectToolData<T>(request: ModelRequest, tool: ChatCompletionMessageToolCall): T {
  const result = resultFor(request, tool);
  if (!result.success) assert.fail(result.error + ": " + result.message);
  return result.data as T;
}

function toolNames(request: ModelRequest) {
  return request.tools?.map((tool) =>
    tool.type === "function" ? tool.function.name : tool.custom.name,
  );
}

function fixture(t: TestContext) {
  const directory = mkdtempSync(join(tmpdir(), "todo-agent-acceptance-"));
  const filename = join(directory, "tasks.json");
  t.after(() => {
    rmSync(filename, { force: true });
    rmSync(filename + ".tmp", { force: true });
    rmdirSync(directory);
  });
  writeFileSync(filename, "[]\n");
  let milliseconds = 0;
  const pendingConfirmations = createPendingConfirmationStore({ now: () => milliseconds });
  const tasks = createJsonTaskStore(filename);
  const bytes = () => readFileSync(filename, "utf8");
  const disk = () => JSON.parse(bytes()) as Task[];
  const advance = (delta: number) => {
    milliseconds += delta;
  };
  const chat = (content: string, options: AgentReplyOptions = {}, history: ChatMessage[] = []) =>
    reply([...history, user(content)], tasks, { pendingConfirmations, ...options });
  return { filename, pendingConfirmations, tasks, bytes, disk, advance, chat };
}
type Fixture = ReturnType<typeof fixture>;

const user = (content: string): ChatMessage => ({ role: "user", content });
const assistant = (content: string, metadata?: ChatMessageMetadata): ChatMessage =>
  metadata ? { role: "assistant", content, metadata } : { role: "assistant", content };
const pendingPrompt = (kind: "duplicate-create" | "cascade-delete" = "cascade-delete") =>
  assistant("请确认旧操作", { kind: "pending_confirmation_prompt", confirmationKind: kind });
const resolved = assistant("确认已处理", { kind: "confirmation_resolved" });
const chatScope = (conversationId: string): ConfirmationScope => ({
  channel: "chat",
  conversationId,
});
const restScope = { channel: "rest" } as const;
const noPending = "当前没有有效的待确认操作，可能已经过期或会话已重置，请重新发起。";

function family(state: Fixture) {
  return state.tasks.createMany([
    { title: "准备面试", subtasks: [{ title: "改简历" }, { title: "复习" }] },
  ]);
}

function startCascade(state: Fixture, scope: ConfirmationScope, tasks: Task[]) {
  return state.pendingConfirmations.create({
    ...scope,
    kind: "cascade-delete",
    rootTaskIds: [tasks[0].id],
    affectedTaskIds: tasks.map((task) => task.id),
  });
}

async function startHttp(t: TestContext, state: Fixture, handler: ReplyHandler = reply) {
  const server = createApp({
    reply: handler,
    tasks: state.tasks,
    pendingConfirmations: state.pendingConfirmations,
  }).listen(0, "127.0.0.1");
  t.after(() => server[Symbol.asyncDispose]());
  await once(server, "listening");
  const { port } = server.address() as AddressInfo;
  return async function request<T = unknown>(method: string, path: string, body?: unknown) {
    const response = await fetch("http://127.0.0.1:" + port + path, {
      method,
      headers: body === undefined ? undefined : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = response.status === 204 ? undefined : await response.json();
    return { status: response.status, body: data as T, headers: response.headers };
  };
}

const collection = "/api/tasks";
const itemPath = (id: string) => collection + "/" + encodeURIComponent(id);
const confirmationPath = (id: string) => collection + "/confirmations/" + encodeURIComponent(id);
interface RestConfirmation {
  error: string;
  confirmationId: string;
  affectedTasks: Task[];
  message: string;
}
interface ChatResponse {
  reply: string;
  metadata?: ChatMessageMetadata;
}
type ListedTasks = {
  today: string;
  total: number;
  truncated: boolean;
  tasks: (Task & {
    normalizedPriority: { rank: number; label: string };
    subtaskProgress?: { total: number; done: number };
  })[];
};

test("T02 REST API 扩展字段创建与校验", async (t) => {
  const state = fixture(t);
  const request = await startHttp(t, state);
  const initial = state.bytes();
  for (const invalid of [
    { title: " " },
    { title: "长".repeat(201) },
    { title: "截止时间", dueDate: "2026-02-30" },
    { title: "注入 ID", id: "hack" },
    { title: "错误工作量", effort: "high" },
    { title: "错误优先级", priority: "1" },
    { title: "绕过确认", allowDuplicate: true },
  ]) {
    assert.equal((await request("POST", collection, invalid)).status, 400);
  }
  assert.equal(state.bytes(), initial);
  const fields = { title: "面试", dueDate: "2028-02-29", priority: 1, effort: "large" };
  const created = await request<Task>("POST", collection, { ...fields, title: " 面试 " });
  assert.equal(created.status, 201);
  assert.deepEqual(created.body, { id: created.body.id, done: false, ...fields });
  const child = await request<Task>("POST", collection, {
    title: "改简历",
    parentId: created.body.id,
  });
  assert.equal(child.status, 201);
  assert.equal(child.body.parentId, created.body.id);
  assert.deepEqual(state.disk(), [created.body, child.body]);
  const list = await request<Task[]>("GET", collection);
  assert.deepEqual(list.body, state.disk());
  assert.equal(isValidIsoDate(list.headers.get("X-Task-Today")), true);
});

test("T03 PATCH 未传保留、传 null 清除且不留 null 键", async (t) => {
  const state = fixture(t);
  const request = await startHttp(t, state);
  const parent = state.tasks.create("根任务");
  const task = state.tasks.create("交报告", {
    dueDate: "2026-11-26",
    priority: 1,
    effort: "small",
    parentId: parent.id,
  });
  const first = await request<Task>("PATCH", itemPath(task.id), { dueDate: null });
  assert.equal(first.status, 200);
  const { dueDate: _oldDate, ...remaining } = task;
  assert.deepEqual(first.body, remaining);
  assert.deepEqual(state.disk(), [parent, remaining]);
  const before = state.bytes();
  for (const invalid of [{ title: null }, { done: null }, { arbitrary: 1 }]) {
    assert.equal((await request("PATCH", itemPath(task.id), invalid)).status, 400);
  }
  assert.equal(state.bytes(), before);
  const second = await request<Task>("PATCH", itemPath(task.id), {
    priority: null,
    effort: null,
    parentId: null,
  });
  assert.equal(second.status, 200);
  const sparse = { id: task.id, title: task.title, done: false };
  assert.deepEqual(second.body, sparse);
  assert.deepEqual(state.tasks.update(task.id, { done: undefined }), sparse);
  assert.deepEqual(state.disk(), [parent, sparse]);
});

test("T04 TaskStore 批量创建原子性", (t) => {
  const state = fixture(t);
  const [parent, child] = family(state);
  const before = state.bytes();
  assert.throws(
    () => state.tasks.createMany([{ title: "有效" }, { title: "无效", parentId: "missing" }]),
    { code: "INVALID_PARENT_TASK" },
  );
  assert.equal(state.bytes(), before);
  assert.throws(
    () =>
      state.tasks.updateMany([
        { id: child.id, done: true },
        { id: "missing", done: true },
      ]),
    { code: "TASK_NOT_FOUND" },
  );
  assert.throws(() => state.tasks.removeMany([child.id, "missing"]), { code: "TASK_NOT_FOUND" });
  assert.throws(() => state.tasks.removeMany([parent.id, child.id]), {
    code: "INVALID_PARENT_TASK",
  });
  assert.equal(state.bytes(), before);
});

test("T05 禁止三层任务结构与一层父子限制", async (t) => {
  const state = fixture(t);
  const [parent, child] = family(state);
  const request = await startHttp(t, state);
  const initial = state.bytes();
  assert.equal(
    (await request("POST", collection, { title: "孙任务", parentId: child.id })).status,
    400,
  );
  for (const parentId of [parent.id, child.id]) {
    assert.equal((await request("PATCH", itemPath(parent.id), { parentId })).status, 400);
  }
  assert.equal(state.bytes(), initial);
  const grandchild = call("create_tasks", {
    tasks: [{ title: "越层", parentId: parent.id, subtasks: [{ title: "第三层" }] }],
  });
  const forged = call("create_tasks", {
    tasks: [{ title: "伪造", subtasks: [{ title: "外部父 ID", parentId: parent.id }] }],
  });
  const overLimit = call("create_tasks", {
    tasks: [
      {
        title: "根 A",
        subtasks: Array.from({ length: 9 }, (_, index) => ({ title: "子 " + index })),
      },
      { title: "根 B" },
    ],
  });
  const invalid = fakeModel(using(grandchild, forged, overLimit), saying("无法创建。"));
  await state.chat("创建这些任务", { client: invalid.client });
  expectToolError(invalid.requests[1], grandchild, "INVALID_PARENT_TASK");
  expectToolError(invalid.requests[1], forged, "INVALID_ARGUMENTS");
  expectToolError(invalid.requests[1], overLimit, "BATCH_LIMIT_EXCEEDED");
  assert.equal(state.bytes(), initial);
  const create = call("create_tasks", {
    tasks: [{ title: "根任务", subtasks: [{ title: "一层子任务" }] }],
  });
  const valid = fakeModel(using(create), saying("已新增。"));
  await state.chat("创建一个带子任务的根任务", { client: valid.client });
  const [newRoot, newChild] = expectToolData<Task[]>(valid.requests[1], create);
  assert.equal(newChild.parentId, newRoot.id);
  assert.deepEqual(state.disk().slice(-2), [newRoot, newChild]);
});

test("T06 父任务进度计算与两阶段级联删除确认", async (t) => {
  const state = fixture(t);
  const [parent, child] = family(state);
  state.tasks.update(child.id, { done: true });
  const before = state.bytes();
  const list = call("list_tasks", { query: parent.title });
  const remove = call("delete_tasks", { ids: [parent.id] });
  const fake = fakeModel(using(list, remove), saying("请确认。"));
  const metadata: ChatMessageMetadata[] = [];
  const options = {
    client: fake.client,
    conversationId: "s1",
    onMetadata: (value: ChatMessageMetadata) => metadata.push(value),
  };
  await state.chat("删除准备面试", options);
  assert.deepEqual(expectToolData<ListedTasks>(fake.requests[1], list).tasks[0].subtaskProgress, {
    total: 2,
    done: 1,
  });
  expectToolError(fake.requests[1], remove, "CONFIRMATION_REQUIRED");
  assert.deepEqual(metadata, [
    { kind: "pending_confirmation_prompt", confirmationKind: "cascade-delete" },
  ]);
  assert.equal(state.bytes(), before);
  assert.match(await state.chat("确认", options), /已确认并删除/);
  assert.equal(fake.requests.length, 2);
  assert.deepEqual(metadata.at(-1), { kind: "confirmation_resolved" });
  assert.deepEqual(state.disk(), []);
});

test("T07 同名未完成任务两阶段重复创建确认", async (t) => {
  const state = fixture(t);
  const original = state.tasks.create("买牛奶");
  const before = state.bytes();
  const create = call("create_tasks", { tasks: [{ title: "买牛奶" }] });
  const fake = fakeModel(using(create), saying("需要确认。"));
  const options = { client: fake.client, conversationId: "s1" };
  await state.chat("添加买牛奶", options);
  expectToolError(fake.requests[1], create, "CONFIRMATION_REQUIRED");
  assert.equal(state.bytes(), before);
  assert.match(await state.chat("确定", options), /已确认并新增/);
  assert.equal(fake.requests.length, 2);
  assert.deepEqual(
    state.disk().map(({ title, done }) => [title, done]),
    [
      ["买牛奶", false],
      ["买牛奶", false],
    ],
  );
  assert.equal(state.disk()[0].id, original.id);
  assert.notEqual(state.disk()[1].id, original.id);
});

test("T08 多轮 Tool Calling 完成官方 6 步验收对话闭环", async (t) => {
  const state = fixture(t);
  const history: ChatMessage[] = [];
  async function chat(content: string, fake: ReturnType<typeof fakeModel>) {
    const answer = await state.chat(
      content,
      { client: fake.client, conversationId: "s1" },
      history,
    );
    history.push(user(content), assistant(answer));
    return answer;
  }
  const createMilk = call("create_tasks", { tasks: [{ title: "买牛奶" }] });
  const first = fakeModel(using(createMilk), saying("已新增买牛奶。"));
  await chat("帮我记一下买牛奶", first);
  const [milk] = expectToolData<Task[]>(first.requests[1], createMilk);
  assert.deepEqual(state.disk(), [milk]);
  const createExpress = call("create_tasks", { tasks: [{ title: "取快递" }] });
  const second = fakeModel(using(createExpress), saying("已新增取快递。"));
  await chat("再记一条取快递", second);
  const [express] = expectToolData<Task[]>(second.requests[1], createExpress);
  assert.deepEqual(state.disk(), [milk, express]);
  const all = call("list_tasks", {});
  const third = fakeModel(using(all), saying("买牛奶和取快递均待处理。"));
  await chat("我有哪些任务？", third);
  assert.deepEqual(
    expectToolData<ListedTasks>(third.requests[1], all).tasks.map(({ done }) => done),
    [false, false],
  );
  const queryMilk = call("list_tasks", { query: "牛奶" });
  const updateMilk = call("update_tasks", { updates: [{ id: milk.id, done: true }] });
  const fourth = fakeModel(using(queryMilk), using(updateMilk), saying("已完成买牛奶。"));
  await chat("牛奶买好了", fourth);
  expectToolData(fourth.requests[2], updateMilk);
  assert.deepEqual(state.disk(), [{ ...milk, done: true }, express]);
  const queryExpress = call("list_tasks", { query: "取快递" });
  const removeExpress = call("delete_tasks", { ids: [express.id] });
  const fifth = fakeModel(using(queryExpress), using(removeExpress), saying("已删除取快递。"));
  await chat("取快递这条删掉吧", fifth);
  expectToolData(fifth.requests[2], removeExpress);
  const finalList = call("list_tasks", {});
  const sixth = fakeModel(using(finalList), saying("已完成：买牛奶。"));
  await chat("再给我看看清单", sixth);
  assert.deepEqual(
    expectToolData<ListedTasks>(sixth.requests[1], finalList).tasks.map(({ title, done }) => ({
      title,
      done,
    })),
    [{ title: "买牛奶", done: true }],
  );
  assert.deepEqual(state.disk(), [{ ...milk, done: true }]);
});

test("T09 模糊匹配多候选消歧与确认后重核验 ID", async (t) => {
  const state = fixture(t);
  const pure = state.tasks.create("买纯牛奶");
  const oat = state.tasks.create("买燕麦牛奶");
  const before = state.bytes();
  const query = call("list_tasks", { query: "牛奶" });
  const guess = call("update_tasks", { updates: [{ id: pure.id, done: true }] });
  const first = fakeModel(using(query, guess), saying("请选第一个或第二个。"));
  const question = await state.chat("牛奶买好了", { client: first.client });
  assert.equal(expectToolData<ListedTasks>(first.requests[1], query).total, 2);
  expectToolError(first.requests[1], guess, "CURRENT_TURN_INTENT_MISMATCH");
  assert.match(question, /多条/);
  assert.match(question, /1\. 买纯牛奶\n2\. 买燕麦牛奶/);
  assert.equal(state.bytes(), before);

  state.tasks.remove(pure.id);
  const replacement = state.tasks.create(pure.title);
  const old = call("update_tasks", { updates: [{ id: pure.id, done: true }] });
  const requery = call("list_tasks", { query: replacement.title });
  const update = call("update_tasks", { updates: [{ id: replacement.id, done: true }] });
  const second = fakeModel(using(old, requery, update), saying("已标记买纯牛奶。"));
  await state.chat("第一个", { client: second.client }, [user("牛奶买好了"), assistant(question)]);
  expectToolError(second.requests[1], old, "TASK_NOT_FOUND");
  assert.equal(
    expectToolData<ListedTasks>(second.requests[1], requery).tasks[0].id,
    replacement.id,
  );
  expectToolData(second.requests[1], update);
  assert.deepEqual(state.disk(), [oat, { ...replacement, done: true }]);
});

test("T10 Tool 参数错误、ID 不存在、5 轮/10 次上限保护", async (t) => {
  const state = fixture(t);
  const p = state.tasks.create("现有任务");
  const before = state.bytes();
  const date = call("create_tasks", { tasks: [{ title: "延期", dueDate: "2025-02-29" }] });
  const missing = call("update_tasks", { updates: [{ id: "no-such-id", done: true }] });
  const noFields = call("update_tasks", { updates: [{ id: p.id }] });
  const reversed = call("list_tasks", { dueDateFrom: "2026-11-26", dueDateTo: "2026-11-25" });
  const invalid = fakeModel(using(date, missing, noFields, reversed), saying("参数无效。"));
  await state.chat("处理这些请求", { client: invalid.client });
  expectToolError(invalid.requests[1], date, "INVALID_ARGUMENTS");
  expectToolError(invalid.requests[1], missing, "TASK_NOT_FOUND");
  expectToolError(invalid.requests[1], noFields, "INVALID_ARGUMENTS");
  expectToolError(invalid.requests[1], reversed, "INVALID_ARGUMENTS");
  assert.equal(state.bytes(), before);

  const rounds = fakeModel(...Array.from({ length: 6 }, () => using(call("list_tasks", {}))));
  const truncated = await state.chat("请看看清单", { client: rounds.client });
  assert.equal(rounds.requests.length, 5);
  assert.match(truncated, /现有任务/);
  assert.match(truncated, /工具调用上限/);
  assert.equal(state.bytes(), before);

  const batch = Array.from({ length: 12 }, (_, i) =>
    call("create_tasks", { tasks: [{ title: "新任务 " + i }] }),
  );
  const cap = fakeModel(using(...batch.slice(0, 6)), using(...batch.slice(6)), saying("不应调用"));
  assert.match(await state.chat("添加一批任务", { client: cap.client }), /工具调用上限/);
  assert.equal(cap.requests.length, 2);
  assert.deepEqual(
    state
      .disk()
      .slice(1)
      .map(({ title }) => title),
    Array.from({ length: 10 }, (_, i) => "新任务 " + i),
  );
});

test("T11 normalizePriority、时区锚点、本周边界与逾期排除已完成", async (t) => {
  const state = fixture(t);
  const legacy = {
    id: "a",
    title: "旧数字优先级",
    priority: 1,
    done: false,
    dueDate: "2026-03-08",
  };
  writeFileSync(state.filename, JSON.stringify([legacy]));
  const high = state.tasks.create("高优先级", { priority: "high", dueDate: "2026-03-07" });
  const low = state.tasks.create("低优先级", { priority: "low", dueDate: "2026-03-04" });
  const completed = state.tasks.create("已完成的过期任务", {
    priority: "high",
    dueDate: "2026-03-01",
  });
  state.tasks.update(completed.id, { done: true });
  const today = state.tasks.create("当天到期", { dueDate: "2026-03-09" });
  state.tasks.create("本周到期", { dueDate: "2026-03-15" });
  const before = state.bytes();
  assert.deepEqual(normalizePriority(1), { rank: 3, label: "高" });
  assert.deepEqual(normalizePriority(1), normalizePriority("high"));
  assert.deepEqual(normalizePriority(4), { rank: 0, label: "4" });
  assert.deepEqual(normalizePriority(null), { rank: -1, label: "" });

  const overdue = call("list_tasks", { overdue: true });
  const dueToday = call("list_tasks", { dueToday: true });
  const limited = call("list_tasks", { priority: "high", limit: 1 });
  const current = fakeModel(using(overdue, dueToday, limited), saying("查询完成。"));
  const now = () => new Date("2026-03-08T16:20:00Z");
  await state.chat("今天有哪些任务到期？", {
    now,
    timeZone: "Asia/Shanghai",
    client: current.client,
  });
  const shanghai = current.requests[0].messages[0].content;
  assert.match(String(shanghai), /2026-03-09，星期一/);
  assert.match(String(shanghai), /明天 2026-03-10，后天 2026-03-11/);
  assert.match(String(shanghai), /2026-03-09 至 2026-03-15/);
  const late = expectToolData<ListedTasks>(current.requests[1], overdue);
  assert.deepEqual(
    late.tasks.map(({ id }) => id),
    [high.id, legacy.id, low.id],
  );
  assert.deepEqual(
    late.tasks.slice(0, 2).map(({ normalizedPriority }) => normalizedPriority.rank),
    [3, 3],
  );
  assert.deepEqual(
    expectToolData<ListedTasks>(current.requests[1], dueToday).tasks.map(({ id }) => id),
    [today.id],
  );
  const data = expectToolData<ListedTasks>(current.requests[1], limited);
  assert.equal(data.total, 3);
  assert.equal(data.tasks.length, 1);
  assert.equal(data.truncated, true);

  const utc = fakeModel(using(dueToday), saying("UTC 日期查询完成。"));
  await state.chat("今天有哪些任务到期？", { now, timeZone: "UTC", client: utc.client });
  assert.match(String(utc.requests[0].messages[0].content), /2026-03-02 至 2026-03-08/);
  assert.deepEqual(
    expectToolData<ListedTasks>(utc.requests[1], dueToday).tasks.map(({ id }) => id),
    [legacy.id],
  );
  assert.equal(state.bytes(), before);
});

test("T12 无 API Key 离线基础模式与复杂语义澄清", async (t) => {
  const state = fixture(t);
  assert.equal(process.env.DEEPSEEK_API_KEY, undefined);
  const conversationId = "offline";
  const opts = { conversationId };
  assert.match(await state.chat("帮我记一下买牛奶", opts), /已新增/);
  assert.match(await state.chat("再记一条取快递", opts), /已新增/);
  const initial = await state.chat("我有哪些任务？", opts);
  assert.match(initial, /待处理/);
  assert.match(initial, /买牛奶/);
  assert.match(initial, /取快递/);
  assert.match(await state.chat("牛奶买好了", opts), /已标记完成/);
  assert.match(await state.chat("取快递这条删掉吧", opts), /已删除/);
  const final = await state.chat("再给我看看清单", opts);
  assert.match(final, /已完成/);
  assert.doesNotMatch(final, /取快递/);
  assert.deepEqual(
    state.disk().map(({ title, done }) => ({ title, done })),
    [{ title: "买牛奶", done: true }],
  );
  assert.match(await state.chat("把买牛奶改名为买面包", opts), /已更新/);
  assert.match(await state.chat("把买面包标记为未完成", opts), /已恢复/);
  assert.deepEqual(
    state.disk().map(({ title, done }) => ({ title, done })),
    [{ title: "买面包", done: false }],
  );
  state.tasks.create("今天交报告", { dueDate: "2026-11-26" });
  const today = await state.chat("今天有哪些任务到期？", {
    ...opts,
    now: () => new Date("2026-11-25T18:00:00Z"),
    timeZone: "Asia/Shanghai",
  });
  assert.match(today, /今天交报告/);
  assert.doesNotMatch(today, /买面包/);
  const before = state.bytes();
  assert.match(
    await state.chat("帮我记一下买牛奶，顺便按刚才说的把其他任务处理掉", opts),
    /请明确/,
  );
  assert.match(await state.chat("帮我记一下买牛奶和取快递", opts), /请明确/);
  assert.equal(state.bytes(), before);
});

test("T13 已执行写 Tool 后模型断网：严禁重放", async (t) => {
  const state = fixture(t);
  const create = call("create_tasks", { tasks: [{ title: "买牛奶" }] });
  const timeout = Object.assign(new Error("Request timed out"), { code: "ETIMEDOUT" });
  const fake = fakeModel(using(create), throwing(timeout));
  const answer = await state.chat("帮我记一下买牛奶", { client: fake.client });
  assert.equal(fake.requests.length, 2);
  const written = expectToolData<Task[]>(fake.requests[1], create);
  assert.match(answer, /已保存/);
  assert.match(answer, /网络异常/);
  assert.deepEqual(state.disk(), written);
  assert.deepEqual(
    state.disk().map(({ title }) => title),
    ["买牛奶"],
  );
});

function httpError(status: number, message: string, code?: string) {
  return Object.assign(new Error(message), { status, code });
}

test("T14 模型与网络全量错误分类矩阵", async (t) => {
  const state = fixture(t);
  const pristine = state.bytes();
  const fallback = "backup-for-" + model;
  process.env.DEEPSEEK_FALLBACK_MODEL = fallback;
  for (const [status, pattern] of [
    [401, /API Key 无效/],
    [402, /余额不足/],
    [403, /缺少访问权限/],
    [429, /限流/],
    [400, /拒绝了请求/],
  ] as const) {
    const fake = fakeModel(throwing(httpError(status, "Invalid model parameter not found")));
    assert.match(await state.chat("添加不会写入的任务", { client: fake.client }), pattern);
    assert.deepEqual(
      fake.requests.map((request) => request.model),
      [model],
    );
    assert.equal(state.bytes(), pristine);
  }

  delete process.env.DEEPSEEK_FALLBACK_MODEL;
  const notFound = httpError(400, "Missing configuration", "model_not_found");
  const noBackup = fakeModel(throwing(notFound));
  assert.match(
    await state.chat("添加不会写入的任务", { client: noBackup.client }),
    /找不到配置的模型/,
  );
  assert.equal(noBackup.requests.length, 1);
  process.env.DEEPSEEK_FALLBACK_MODEL = fallback;
  const badBackup = fakeModel(throwing(notFound), throwing(httpError(429, "Too many requests")));
  assert.match(await state.chat("添加不会写入的任务", { client: badBackup.client }), /限流/);
  assert.deepEqual(
    badBackup.requests.map((request) => request.model),
    [model, fallback],
  );
  const notFoundTwice = fakeModel(throwing(notFound), throwing(notFound));
  assert.match(
    await state.chat("添加不会写入的任务", { client: notFoundTwice.client }),
    /找不到配置的模型/,
  );
  assert.deepEqual(
    notFoundTwice.requests.map((request) => request.model),
    [model, fallback],
  );
  assert.equal(state.bytes(), pristine);

  const retryable = [
    httpError(503, "Service unavailable"),
    Object.assign(new Error("fetch failed"), { cause: { code: "ECONNRESET" } }),
    Object.assign(new Error("Request timed out"), { code: "ETIMEDOUT" }),
  ];
  for (const [index, error] of retryable.entries()) {
    const ambiguous = fakeModel(throwing(error));
    const before = state.bytes();
    const answer = await state.chat("按刚才说的做，细节你自己决定", { client: ambiguous.client });
    assert.match(answer, /服务暂时异常|网络异常/);
    assert.doesNotMatch(answer, /离线处理结果/);
    assert.equal(state.bytes(), before);
    const fake = fakeModel(throwing(error));
    const title = "离线任务 " + index;
    assert.match(await state.chat("添加" + title, { client: fake.client }), /离线处理结果/);
    assert.deepEqual(
      fake.requests.map((request) => request.model),
      [model],
    );
    assert.equal(state.disk().filter((task) => task.title === title).length, 1);
  }
});

test("T16 模型不能通过确认布尔值绕过用户确认", async (t) => {
  const state = fixture(t);
  const [parent] = family(state);
  state.tasks.create("买牛奶");
  const before = state.bytes();
  const duplicate = call("create_tasks", { tasks: [{ title: "买牛奶" }], allowDuplicate: true });
  const nested = call("create_tasks", { tasks: [{ title: "买牛奶", allowDuplicate: true }] });
  const remove = call("delete_tasks", { ids: [parent.id], confirmCascade: true });
  const fake = fakeModel(using(duplicate, nested, remove), saying("无效。"));
  await state.chat("处理这些任务", { client: fake.client, conversationId: "s1" });
  for (const invalid of [duplicate, nested, remove]) {
    expectToolError(fake.requests[1], invalid, "INVALID_ARGUMENTS");
  }
  assert.doesNotMatch(JSON.stringify(fake.requests[0].tools), /allowDuplicate|confirmCascade/);
  assert.equal(state.pendingConfirmations.getByConversationId("s1"), undefined);
  assert.equal(state.bytes(), before);
});

test("T17 重复创建必须经过服务端 PendingConfirmationStore 两阶段确认", async (t) => {
  const state = fixture(t);
  const milk = state.tasks.create("买牛奶");
  const before = state.bytes();
  const create = call("create_tasks", { tasks: [{ title: milk.title }] });
  const laterWrite = call("create_tasks", { tasks: [{ title: "另一条任务" }] });
  const fake = fakeModel(using(create, laterWrite), saying("请确认。"));
  const conversationId = "s1";
  await state.chat("添加买牛奶", { conversationId, client: fake.client });
  expectToolError(fake.requests[1], create, "CONFIRMATION_REQUIRED");
  expectToolError(fake.requests[1], laterWrite, "PENDING_CONFIRMATION_EXISTS");
  const pending = state.pendingConfirmations.getByConversationId(conversationId);
  assert.equal(pending?.kind, "duplicate-create");
  assert.deepEqual(pending.duplicateTaskIds, [milk.id]);
  assert.deepEqual(pending.duplicateTitles, [milk.title]);
  assert.equal(state.bytes(), before);

  const injected = call("update_tasks", { updates: [{ id: milk.id, done: true }] });
  const readOnly = fakeModel(using(injected), saying("正在等待。"));
  await state.chat("我确认一下", { conversationId, client: readOnly.client });
  assert.deepEqual(toolNames(readOnly.requests[0]), ["list_tasks"]);
  expectToolError(readOnly.requests[1], injected, "PENDING_CONFIRMATION_EXISTS");
  assert.equal(
    state.pendingConfirmations.getById(pending.confirmationId)?.kind,
    "duplicate-create",
  );
  assert.equal(state.bytes(), before);
});

test("T18 父任务级联删除必须经过服务端 PendingConfirmationStore 两阶段确认", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const before = state.bytes();
  const conversationId = "s1";
  const list = call("list_tasks", {});
  const remove = call("delete_tasks", { ids: entries.map((task) => task.id) });
  const add = call("create_tasks", { tasks: [{ title: "未授权任务" }] });
  const fake = fakeModel(using(list, remove, add), saying("等待确认。"));
  await state.chat("面试父任务和子任务都不用了", { conversationId, client: fake.client });
  expectToolError(fake.requests[1], remove, "CONFIRMATION_REQUIRED");
  expectToolError(fake.requests[1], add, "PENDING_CONFIRMATION_EXISTS");
  const pending = state.pendingConfirmations.getByConversationId(conversationId);
  assert.equal(pending?.kind, "cascade-delete");
  assert.deepEqual(pending.rootTaskIds, [entries[0].id]);
  assert.deepEqual(
    pending.affectedTaskIds,
    entries.map((task) => task.id),
  );
  assert.equal(state.bytes(), before);
});

test("T19 确认意图精确匹配、Store 越权隔离、跨会话聊天提示与待确认期间受限只读查询", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const before = state.bytes();
  const chat = startCascade(state, chatScope("s1"), entries);
  const rest = startCascade(state, restScope, entries);
  for (const action of [state.pendingConfirmations.consume, state.pendingConfirmations.cancel]) {
    assert.deepEqual(action(chat.confirmationId, chatScope("s2")), {
      ok: false,
      reason: "FORBIDDEN",
    });
    assert.deepEqual(action(chat.confirmationId, restScope), { ok: false, reason: "FORBIDDEN" });
    assert.deepEqual(action(rest.confirmationId, chatScope("s1")), {
      ok: false,
      reason: "FORBIDDEN",
    });
  }
  assert.equal(state.pendingConfirmations.getById(rest.confirmationId)?.channel, "rest");
  assert.equal(
    state.pendingConfirmations.getByConversationId("s1")?.confirmationId,
    chat.confirmationId,
  );
  const unused = fakeModel();
  assert.equal(
    await state.chat("确认", { conversationId: "s2", client: unused.client }),
    noPending,
  );
  assert.equal(unused.requests.length, 0);
  assert.equal(state.bytes(), before);
  for (const vague of ["第一个", "我确认一下", "不确定", "我确认删除，但先让我看看清单"]) {
    assert.equal(parseConfirmationIntent(vague), "unclear");
    const fake = fakeModel(saying("仍等待确认。"));
    await state.chat(vague, { conversationId: "s1", client: fake.client });
    assert.deepEqual(toolNames(fake.requests[0]), ["list_tasks"]);
    assert.equal(state.bytes(), before);
  }
  const read = call("list_tasks", {});
  const fake = fakeModel(using(read), saying("当前有三个任务。"));
  const metadata: ChatMessageMetadata[] = [];
  const options = {
    conversationId: "s1",
    client: fake.client,
    onMetadata: (value: ChatMessageMetadata) => metadata.push(value),
  };
  assert.match(await state.chat("当前有哪些任务", options), /当前有三个任务/);
  assert.deepEqual(toolNames(fake.requests[0]), ["list_tasks"]);
  assert.equal(expectToolData<ListedTasks>(fake.requests[1], read).total, 3);
  assert.deepEqual(metadata, []);
  assert.equal(
    state.pendingConfirmations.getByConversationId("s1")?.confirmationId,
    chat.confirmationId,
  );
  assert.equal(state.bytes(), before);
  assert.match(await state.chat("取消", options), /已取消/);
  assert.equal(state.pendingConfirmations.getById(chat.confirmationId), undefined);
  assert.deepEqual(metadata, [{ kind: "confirmation_resolved" }]);
  assert.equal(state.bytes(), before);
  startCascade(state, chatScope("s1"), entries);
  assert.match(await state.chat("确认。", options), /已确认并删除/);
  assert.equal(fake.requests.length, 2);
  assert.deepEqual(state.disk(), []);
});

test("T20 精确匹配边界、无待确认口语不误判、过期/重置提示、重复消费与目标变化拦截", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const before = state.bytes();
  assert.equal(parseConfirmationIntent(" 确认。！？ "), "confirm");
  assert.equal(parseConfirmationIntent("确认 。"), "unclear");
  assert.equal(parseConfirmationIntent("算了"), "cancel");
  assert.equal(parseConfirmationIntent("不用了，先看看"), "unclear");
  assert.equal(parseStaleConfirmationIntent("取消删除。"), "cancel");
  assert.equal(parseStaleConfirmationIntent("我确认一下"), "none");
  for (const acknowledgement of ["好的", "对", "可以", "算了"]) {
    assert.equal(parseStaleConfirmationIntent(acknowledgement), "none");
    const fake = fakeModel(saying("收到。"));
    assert.equal(await state.chat(acknowledgement, { client: fake.client }), "收到。");
    assert.equal(fake.requests.length, 1);
    assert.equal(state.bytes(), before);
  }
  const expired = startCascade(state, chatScope("expired"), entries);
  state.advance(600_000);
  const unused = fakeModel();
  for (const phrase of ["确认", "取消", "确认删除"]) {
    for (const options of [
      { conversationId: "expired" },
      { conversationId: "reloaded" },
      { conversationId: "expired", pendingConfirmations: createPendingConfirmationStore() },
    ]) {
      assert.equal(await state.chat(phrase, { ...options, client: unused.client }), noPending);
      assert.equal(state.bytes(), before);
    }
  }
  assert.deepEqual(
    state.pendingConfirmations.consume(expired.confirmationId, chatScope("expired")),
    {
      ok: false,
      reason: "EXPIRED",
    },
  );

  startCascade(state, chatScope("s1"), entries);
  const options = { conversationId: "s1", client: unused.client };
  assert.match(await state.chat("算了", options), /已取消/);
  assert.equal(state.bytes(), before);
  const staleTarget = startCascade(state, chatScope("s1"), entries);
  state.tasks.create("新子任务", { parentId: entries[0].id });
  const changed = state.bytes();
  assert.match(await state.chat("确认", options), /变化.*重新/);
  assert.deepEqual(
    state.pendingConfirmations.consume(staleTarget.confirmationId, chatScope("s1")),
    {
      ok: false,
      reason: "NOT_FOUND",
    },
  );
  assert.equal(state.bytes(), changed);

  const valid = startCascade(state, chatScope("s1"), state.disk());
  assert.match(await state.chat("好的", options), /已确认并删除/);
  assert.deepEqual(state.disk(), []);
  assert.deepEqual(state.pendingConfirmations.consume(valid.confirmationId, chatScope("s1")), {
    ok: false,
    reason: "NOT_FOUND",
  });
  assert.equal(unused.requests.length, 0);
});

test("T21 REST API 级联删除两阶段确认与取消端点", async (t) => {
  const state = fixture(t);
  const request = await startHttp(t, state);
  const entries = family(state);
  const before = state.bytes();
  const first = await request<RestConfirmation>("DELETE", itemPath(entries[0].id));
  assert.equal(first.status, 409);
  assert.equal(first.body.error, "CONFIRMATION_REQUIRED");
  assert.deepEqual(first.body.affectedTasks, entries);
  assert.equal(state.bytes(), before);
  const path = confirmationPath(first.body.confirmationId);
  const concurrent = await Promise.all([request("POST", path), request("POST", path)]);
  assert.deepEqual(concurrent.map(({ status }) => status).sort(), [204, 404]);
  assert.deepEqual(state.disk(), []);
  assert.equal((await request("POST", path)).status, 404);

  const newEntries = family(state);
  const next = await request<RestConfirmation>("DELETE", itemPath(newEntries[0].id));
  assert.equal(next.status, 409);
  assert.notEqual(next.body.confirmationId, first.body.confirmationId);
  const cancelPath = confirmationPath(next.body.confirmationId);
  const preserved = state.bytes();
  assert.equal((await request("DELETE", cancelPath)).status, 204);
  assert.equal((await request("POST", cancelPath)).status, 404);
  assert.equal(state.bytes(), preserved);

  const chat = startCascade(state, chatScope("s1"), newEntries);
  for (const method of ["POST", "DELETE"]) {
    const wrongChannel = await request<{ error: string }>(
      method,
      confirmationPath(chat.confirmationId),
    );
    assert.equal(wrongChannel.status, 404);
    assert.equal(wrongChannel.body.error, "CONFIRMATION_NOT_FOUND");
  }
  assert.equal(
    state.pendingConfirmations.getByConversationId("s1")?.confirmationId,
    chat.confirmationId,
  );
  const expires = await request<RestConfirmation>("DELETE", itemPath(newEntries[0].id));
  const expiresCancel = await request<RestConfirmation>("DELETE", itemPath(newEntries[0].id));
  state.advance(600_000);
  const expired = await request<{ error: string }>(
    "POST",
    confirmationPath(expires.body.confirmationId),
  );
  const expiredCancel = await request<{ error: string }>(
    "DELETE",
    confirmationPath(expiresCancel.body.confirmationId),
  );
  assert.deepEqual([expired.status, expiredCancel.status], [409, 409]);
  assert.deepEqual(
    [expired.body.error, expiredCancel.body.error],
    ["CONFIRMATION_EXPIRED", "CONFIRMATION_EXPIRED"],
  );
  assert.equal(state.bytes(), preserved);

  const snapshot = await request<RestConfirmation>("DELETE", itemPath(newEntries[0].id));
  state.tasks.remove(newEntries[1].id);
  state.tasks.create("改后的子任务", { parentId: newEntries[0].id });
  const changed = state.bytes();
  const changedConfirm = await request<{ error: string }>(
    "POST",
    confirmationPath(snapshot.body.confirmationId),
  );
  assert.equal(changedConfirm.status, 409);
  assert.equal(changedConfirm.body.error, "CONFIRMATION_TARGET_CHANGED");
  assert.equal(state.bytes(), changed);
});

test("T22 ?cascade=true 不能绕过 REST 服务端确认", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const before = state.bytes();
  const request = await startHttp(t, state);
  const attempt = await request<RestConfirmation>(
    "DELETE",
    itemPath(entries[0].id) + "?cascade=true",
  );
  assert.equal(attempt.status, 409);
  assert.equal(attempt.body.error, "CONFIRMATION_REQUIRED");
  assert.deepEqual(attempt.body.affectedTasks, entries);
  assert.equal(state.bytes(), before);
  assert.equal(state.pendingConfirmations.getById(attempt.body.confirmationId)?.channel, "rest");
});

test("T23 备用模型跨轮次保持不切回主模型", async (t) => {
  const state = fixture(t);
  const backup = "backup-m-" + model;
  process.env.DEEPSEEK_FALLBACK_MODEL = backup;
  const create = call("create_tasks", { tasks: [{ title: "买牛奶" }] });
  const fake = fakeModel(
    throwing(httpError(400, "Model Not Exist")),
    using(create),
    saying("已记下买牛奶。"),
  );
  assert.match(await state.chat("添加买牛奶", { client: fake.client }), /已记下/);
  assert.deepEqual(
    fake.requests.map((request) => request.model),
    [model, backup, backup],
  );
  assert.deepEqual(state.disk(), expectToolData<Task[]>(fake.requests[2], create));
  const messages = fake.requests[2].messages;
  assert(
    messages.some(
      (message) =>
        message.role === "assistant" && message.tool_calls?.some((tool) => tool.id === create.id),
    ),
  );
  assert(messages.some((message) => message.role === "tool" && message.tool_call_id === create.id));
});

test("T24 聊天请求 conversationId 兼容旧请求与无会话确认拦截", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const create = call("create_tasks", { tasks: [{ title: "买牛奶" }] });
  const query = call("list_tasks", {});
  const listParent = call("list_tasks", { query: entries[0].title });
  const remove = call("delete_tasks", { ids: [entries[0].id] });
  const fake = fakeModel(
    using(create),
    saying("已新增买牛奶。"),
    using(query),
    saying("当前的任务清单。"),
    using(create),
    saying("需要确认。"),
    using(listParent, remove),
    saying("需要确认。"),
    using(create),
    saying("请回复确认。"),
  );
  const request = await startHttp(t, state, (messages, tasks, options) =>
    reply(messages, tasks, { ...options, client: fake.client }),
  );
  const endpoint = "/api/chat";
  const response = await request<ChatResponse>("POST", endpoint, {
    messages: [user("添加买牛奶")],
  });
  assert.equal(response.status, 200);
  assert.deepEqual(response.body, { reply: "已新增买牛奶。" });
  const [milk] = expectToolData<Task[]>(fake.requests[1], create);
  assert.deepEqual(state.disk(), [...entries, milk]);
  const list = await request<ChatResponse>("POST", endpoint, {
    messages: [user("我有哪些任务？")],
  });
  assert.deepEqual(list.body, { reply: "当前的任务清单。" });
  const before = state.bytes();
  await request("POST", endpoint, { messages: [user("添加买牛奶")] });
  expectToolError(fake.requests[5], create, "CONFIRMATION_SESSION_REQUIRED");
  await request("POST", endpoint, { messages: [user("删除准备面试")] });
  expectToolError(fake.requests[7], remove, "CONFIRMATION_SESSION_REQUIRED");
  assert.equal(state.bytes(), before);

  const conversationId = randomUUID();
  assert.equal(
    (await request("POST", endpoint, { conversationId: " ", messages: [user("你好")] })).status,
    400,
  );
  assert.equal(
    (
      await request("POST", endpoint, {
        conversationId,
        messages: [{ ...user("你好"), metadata: { injected: true } }],
      })
    ).status,
    400,
  );
  const pending = await request<ChatResponse>("POST", endpoint, {
    conversationId,
    messages: [user("添加买牛奶")],
  });
  assert.equal(pending.status, 200);
  assert.deepEqual(pending.body.metadata, {
    kind: "pending_confirmation_prompt",
    confirmationKind: "duplicate-create",
  });
  assert.equal(state.bytes(), before);
  const confirmed = await request<ChatResponse>("POST", endpoint, {
    conversationId,
    messages: [
      user("添加买牛奶"),
      assistant(pending.body.reply, pending.body.metadata),
      user("确认"),
    ],
  });
  assert.equal(confirmed.status, 200);
  assert.deepEqual(confirmed.body.metadata, { kind: "confirmation_resolved" });
  assert.equal(fake.requests.length, 10);
  assert.equal(state.disk().filter((task) => task.title === "买牛奶").length, 2);
});

test("T26-A 待确认过期后普通肯定语不得恢复旧操作", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const before = state.bytes();
  const conversationId = "s1";
  startCascade(state, chatScope(conversationId), entries);
  state.advance(600_001);
  const fake = fakeModel();
  const metadata: ChatMessageMetadata[] = [];
  const options = {
    conversationId,
    client: fake.client,
    onMetadata: (value: ChatMessageMetadata) => metadata.push(value),
  };
  assert.match(
    await state.chat("好的", options, [user("删除准备面试"), pendingPrompt()]),
    /已失效/,
  );
  assert.equal(fake.requests.length, 0);
  assert.deepEqual(metadata, []);
  assert.equal(state.bytes(), before);
});

test("T26-B 服务重启后历史确认提示上下文保护", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const before = state.bytes();
  const conversationId = "s1";
  const old = startCascade(state, chatScope(conversationId), entries);
  const restarted = createPendingConfirmationStore();
  const fake = fakeModel();
  for (const message of ["对", "算了"]) {
    assert.match(
      await state.chat(
        message,
        {
          conversationId,
          client: fake.client,
          pendingConfirmations: restarted,
        },
        [user("删除准备面试"), pendingPrompt()],
      ),
      /已失效/,
    );
  }
  assert.equal(fake.requests.length, 0);
  assert.equal(restarted.getById(old.confirmationId), undefined);
  assert.equal(state.bytes(), before);
});

test("T26-C 无历史确认提示的普通口语不误拦截", async (t) => {
  const state = fixture(t);
  const before = state.bytes();
  const histories = [
    [user("你是谁"), assistant("你好，我可以帮你确认任务状态。")],
    [user("删除准备面试"), pendingPrompt(), user("取消"), resolved],
  ];
  for (const history of histories) {
    const fake = fakeModel(saying("好的，随时可以继续。"));
    assert.equal(hasRecentUnresolvedConfirmationPrompt(history), false);
    assert.match(await state.chat("好的", { client: fake.client }, history), /随时可以继续/);
    assert.equal(fake.requests.length, 1);
    assert.deepEqual(toolNames(fake.requests[0]), [
      "list_tasks",
      "create_tasks",
      "update_tasks",
      "delete_tasks",
    ]);
    assert(fake.requests[0].messages.every((message) => !("metadata" in message)));
    assert.equal(state.bytes(), before);
  }
});

test("T26-D 过期确认后的普通查询仍可使用且不清除 stale 标记", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const conversationId = "s1";
  const before = state.bytes();
  startCascade(state, chatScope(conversationId), entries);
  state.advance(600_000);
  const list = call("list_tasks", {});
  const fake = fakeModel(using(list), saying("准备面试及两条子任务仍在清单。"));
  const metadata: ChatMessageMetadata[] = [];
  const options = {
    conversationId,
    client: fake.client,
    onMetadata: (value: ChatMessageMetadata) => metadata.push(value),
  };
  const history = [user("删除准备面试"), pendingPrompt()];
  const answer = await state.chat("当前有哪些任务", options, history);
  assert.match(answer, /仍在清单/);
  assert.deepEqual(toolNames(fake.requests[0]), ["list_tasks"]);
  assert.equal(expectToolData<ListedTasks>(fake.requests[1], list).total, 3);
  history.push(user("当前有哪些任务"), assistant(answer));
  assert.equal(hasRecentUnresolvedConfirmationPrompt(history), true);
  assert.match(await state.chat("好的", options, history), /已失效/);
  assert.equal(fake.requests.length, 2);
  assert.deepEqual(metadata, []);
  assert.equal(state.bytes(), before);
});

test("T26-E stale 标记绝对不能授权执行写操作", async (t) => {
  const state = fixture(t);
  const original = state.tasks.create("买牛奶");
  const before = state.bytes();
  const history = [user("添加买牛奶"), pendingPrompt("duplicate-create")];
  const create = call("create_tasks", { tasks: [{ title: original.title }] });
  const noSession = fakeModel(using(create), saying("需要会话。"));
  await state.chat("现在请重新添加买牛奶", { client: noSession.client }, history);
  expectToolError(noSession.requests[1], create, "CONFIRMATION_SESSION_REQUIRED");
  assert.equal(state.bytes(), before);

  const conversationId = "s1";
  const old = state.pendingConfirmations.create({
    kind: "duplicate-create",
    channel: "chat",
    conversationId,
    tasks: [{ title: original.title }],
    duplicateTaskIds: [original.id],
    duplicateTitles: [original.title],
  });
  state.advance(600_000);
  const withSession = fakeModel(using(create), saying("需要确认。"));
  await state.chat("现在请重新添加买牛奶", { conversationId, client: withSession.client }, history);
  expectToolError(withSession.requests[1], create, "CONFIRMATION_REQUIRED");
  const token = state.pendingConfirmations.getByConversationId(conversationId);
  assert(token && token.confirmationId !== old.confirmationId);
  assert.equal(state.pendingConfirmations.getById(old.confirmationId), undefined);
  assert.equal(state.bytes(), before);

  state.tasks.update(original.id, { done: true });
  const changed = state.bytes();
  assert.match(
    await state.chat("确认", { conversationId, client: withSession.client }),
    /变化.*重新/,
  );
  assert.equal(withSession.requests.length, 2);
  assert.equal(state.bytes(), changed);
});

test("T27-A stale 历史确认下的普通查询禁止写操作", async (t) => {
  const state = fixture(t);
  const [parent] = family(state);
  const before = state.bytes();
  const fakeWrite = call("delete_tasks", { ids: [parent.id] });
  const fake = fakeModel(using(fakeWrite), saying("假称已删除。"));
  const history = [user("删除准备面试"), pendingPrompt()];
  const answer = await state.chat(
    "当前有哪些任务",
    { conversationId: "s1", client: fake.client },
    history,
  );
  assert.deepEqual(toolNames(fake.requests[0]), ["list_tasks"]);
  for (const request of fake.requests)
    assert(request.messages.every((message) => !("metadata" in message)));
  expectToolError(fake.requests[1], fakeWrite, "STALE_CONTEXT_READ_ONLY");
  assert.doesNotMatch(answer, /已删除/);
  assert.equal(state.pendingConfirmations.getByConversationId("s1"), undefined);
  assert.equal(state.bytes(), before);
});

test("T27-B stale 历史确认下明确重新发起写操作", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const conversationId = "s1";
  const before = state.bytes();
  const old = startCascade(state, chatScope(conversationId), entries);
  state.advance(600_000);
  const message = "现在请重新删除准备面试";
  const allowedTools = ["list_tasks", "delete_tasks"];
  assert.deepEqual(
    determineTurnToolPolicy(message, { hasRecentUnresolvedConfirmationPrompt: true }),
    {
      mode: "scoped-write",
      allowedTools,
      currentTurnIntent: {
        operation: "delete",
        targetText: "准备面试",
        sourceMessage: message,
        allowedTools,
      },
    },
  );
  const list = call("list_tasks", { query: "准备面试" });
  const remove = call("delete_tasks", { ids: [entries[0].id] });
  const fake = fakeModel(using(list), using(remove), saying("请确认新的操作。"));
  const metadata: ChatMessageMetadata[] = [];
  const answer = await state.chat(
    message,
    {
      conversationId,
      client: fake.client,
      onMetadata: (value) => metadata.push(value),
    },
    [user("旧删除准备面试"), pendingPrompt()],
  );
  assert.match(answer, /请回复「确认」/);
  assert.deepEqual(toolNames(fake.requests[0]), allowedTools);
  assert.deepEqual(fake.requests[0].messages.slice(1), [user(message)]);
  expectToolError(fake.requests[2], remove, "CONFIRMATION_REQUIRED");
  const fresh = state.pendingConfirmations.getByConversationId(conversationId);
  assert(fresh && fresh.confirmationId !== old.confirmationId);
  assert.equal(state.pendingConfirmations.getById(old.confirmationId), undefined);
  assert.deepEqual(metadata, [
    { kind: "pending_confirmation_prompt", confirmationKind: "cascade-delete" },
  ]);
  assert.equal(state.bytes(), before);
});

test("T27-C stale 历史确认下模糊消息与指代词不得触发旧写操作", async (t) => {
  const state = fixture(t);
  const [parent] = family(state);
  const before = state.bytes();
  const history = [user("删除准备面试"), pendingPrompt()];
  for (const message of ["你觉得呢", "按刚才说的做"]) {
    assert.deepEqual(
      determineTurnToolPolicy(message, { hasRecentUnresolvedConfirmationPrompt: true }),
      {
        mode: "read-only",
        allowedTools: ["list_tasks"],
        requiresClarification: true,
      },
    );
    const injected = call("delete_tasks", { ids: [parent.id] });
    const fake = fakeModel(using(injected), saying("误称已删除。"));
    const answer = await state.chat(
      message,
      { conversationId: "s1", client: fake.client },
      history,
    );
    assert.match(answer, /请明确/);
    assert.deepEqual(toolNames(fake.requests[0]), ["list_tasks"]);
    expectToolError(fake.requests[1], injected, "STALE_CONTEXT_READ_ONLY");
    assert.equal(state.pendingConfirmations.getByConversationId("s1"), undefined);
    assert.equal(state.bytes(), before);
  }
});

test("T27-D stale scoped-write 分支的 Tool Call 严格绑定当前轮操作、目标与最新查询 ID", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const [parent] = entries;
  const milk = state.tasks.create("买牛奶");
  const other = state.tasks.create("其他任务");
  const before = state.bytes();
  const conversationId = "s1";
  const old = startCascade(state, chatScope(conversationId), entries);
  state.advance(600_000);
  const history = [user("旧删除准备面试"), pendingPrompt()];
  const oldDelete = call("delete_tasks", { ids: [parent.id] });
  const wrongTitle = call("create_tasks", { tasks: [{ title: "另外的任务" }] });
  const wrongSubtasks = call("create_tasks", {
    tasks: [{ title: milk.title, subtasks: [{ title: other.title }] }],
  });
  const create = call("create_tasks", { tasks: [{ title: milk.title }] });
  const creating = fakeModel(
    using(oldDelete, wrongTitle, wrongSubtasks),
    using(create),
    saying("新的创建待确认。"),
  );
  await state.chat("现在请重新添加买牛奶", { conversationId, client: creating.client }, history);
  for (const tool of [oldDelete, wrongTitle, wrongSubtasks]) {
    expectToolError(creating.requests[1], tool, "CURRENT_TURN_INTENT_MISMATCH");
  }
  expectToolError(creating.requests[2], create, "CONFIRMATION_REQUIRED");
  const freshCreate = state.pendingConfirmations.getByConversationId(conversationId);
  assert(freshCreate && freshCreate.confirmationId !== old.confirmationId);
  assert.equal(state.bytes(), before);
  assert.match(await state.chat("取消", { conversationId, client: creating.client }), /已取消/);
  assert.equal(state.bytes(), before);

  const badCreate = call("create_tasks", { tasks: [{ title: "买牛奶" }] });
  const unqueried = call("delete_tasks", { ids: [parent.id] });
  const listOther = call("list_tasks", { query: other.title });
  const wrongTarget = call("delete_tasks", { ids: [other.id] });
  const listParent = call("list_tasks", { query: parent.title });
  const laterList = call("list_tasks", { query: other.title });
  const staleListId = call("delete_tasks", { ids: [parent.id] });
  const finalList = call("list_tasks", { query: parent.title });
  const authorized = call("delete_tasks", { ids: [parent.id] });
  const deleting = fakeModel(
    using(badCreate, unqueried),
    using(listOther, wrongTarget),
    using(listParent, laterList, staleListId),
    using(finalList, authorized),
    saying("新的删除待确认。"),
  );
  await state.chat("现在请重新删除准备面试", { conversationId, client: deleting.client }, history);
  for (const tool of [badCreate, unqueried, wrongTarget, staleListId]) {
    expectToolError(deleting.requests[4], tool, "CURRENT_TURN_INTENT_MISMATCH");
  }
  expectToolError(deleting.requests[4], authorized, "CONFIRMATION_REQUIRED");
  const freshDelete = state.pendingConfirmations.getByConversationId(conversationId);
  assert(freshDelete && freshDelete.confirmationId !== old.confirmationId);
  assert.equal(state.pendingConfirmations.getById(old.confirmationId), undefined);
  assert.equal(state.bytes(), before);
  await state.chat("取消", { conversationId, client: deleting.client });
  assert.equal(state.bytes(), before);

  const read = call("list_tasks", { query: other.title });
  const unqueriedUpdate = call("update_tasks", { updates: [{ id: other.id, done: true }] });
  const wrongUpdate = call("update_tasks", { updates: [{ id: other.id, done: false }] });
  const authorizedUpdate = call("update_tasks", { updates: [{ id: other.id, done: true }] });
  const updating = fakeModel(
    using(unqueriedUpdate, read, wrongUpdate, authorizedUpdate),
    saying("已标记完成。"),
  );
  await state.chat(
    "把「其他任务」标记为已完成",
    { conversationId, client: updating.client },
    history,
  );
  expectToolError(updating.requests[1], unqueriedUpdate, "CURRENT_TURN_INTENT_MISMATCH");
  expectToolError(updating.requests[1], wrongUpdate, "CURRENT_TURN_INTENT_MISMATCH");
  expectToolData(updating.requests[1], authorizedUpdate);
  assert.deepEqual(state.disk(), [...entries, milk, { ...other, done: true }]);
});

test("R1 正常澄清后肯定回复允许模型查证任务并更新属性", async (t) => {
  const state = fixture(t);
  const history = [user("帮我调整牛奶任务"), assistant("要把买牛奶的优先级设为高吗？")];

  for (const acknowledgement of ["好的", "可以", "是的"]) {
    const milk = state.tasks.create("买牛奶");
    const list = call("list_tasks", { query: milk.title });
    const update = call("update_tasks", { updates: [{ id: milk.id, priority: "high" }] });
    const fake = fakeModel(using(list), using(update), saying("已设为高优先级。"));
    const answer = await state.chat(acknowledgement, { client: fake.client }, history);
    assert.deepEqual(toolNames(fake.requests[0]), [
      "list_tasks",
      "create_tasks",
      "update_tasks",
      "delete_tasks",
    ]);
    expectToolData(fake.requests[2], update);
    assert.equal(answer, "已设为高优先级。");
    assert.deepEqual(state.disk(), [{ ...milk, priority: "high" }]);
    state.tasks.remove(milk.id);
  }
});

test("R2 正常对话支持带修饰语的任务操作，模糊多候选仍要求消歧", async (t) => {
  const state = fixture(t);
  const milk = state.tasks.create("买牛奶");
  const list = call("list_tasks", { query: milk.title });
  const update = call("update_tasks", { updates: [{ id: milk.id, done: true }] });
  const completing = fakeModel(using(list), using(update), saying("已完成。"));
  await state.chat("超市的牛奶买好了", { client: completing.client });
  expectToolData(completing.requests[2], update);
  assert.deepEqual(state.disk(), [{ ...milk, done: true }]);

  const remove = call("delete_tasks", { ids: [milk.id] });
  const deleting = fakeModel(using(list), using(remove), saying("已删除。"));
  await state.chat("把昨天那个买牛奶删掉", { client: deleting.client });
  expectToolData(deleting.requests[2], remove);
  assert.deepEqual(state.disk(), []);

  const pure = state.tasks.create("买纯牛奶");
  const oat = state.tasks.create("买燕麦牛奶");
  const before = state.bytes();
  const listOnlyOne = call("list_tasks", { query: pure.title });
  const guess = call("update_tasks", { updates: [{ id: pure.id, done: true }] });
  const ambiguous = fakeModel(using(listOnlyOne, guess), saying("假称已完成。"));
  const question = await state.chat("牛奶买好了", { client: ambiguous.client });
  expectToolError(ambiguous.requests[1], guess, "CURRENT_TURN_INTENT_MISMATCH");
  assert.match(question, /1\. 买纯牛奶\n2\. 买燕麦牛奶/);
  assert.equal(state.bytes(), before);

  const listSelected = call("list_tasks", { query: oat.title });
  const selected = call("update_tasks", { updates: [{ id: oat.id, done: true }] });
  const disambiguated = fakeModel(using(listSelected, selected), saying("已完成买燕麦牛奶。"));
  await state.chat("第二个", { client: disambiguated.client }, [
    user("牛奶买好了"),
    assistant(question),
  ]);
  expectToolData(disambiguated.requests[1], selected);
  assert.deepEqual(state.disk(), [pure, { ...oat, done: true }]);
});

test("R3 正常对话保存多次查询与同轮更新快照，失效历史下的复合指令仍只读", async (t) => {
  const state = fixture(t);
  const milk = state.tasks.create("买牛奶");
  const parcel = state.tasks.create("取快递");
  const listMilk = call("list_tasks", { query: milk.title });
  const listParcel = call("list_tasks", { query: parcel.title });
  const update = call("update_tasks", { updates: [{ id: milk.id, done: true }] });
  const remove = call("delete_tasks", { ids: [parcel.id] });
  const compound = "把买牛奶标为完成并把取快递删掉";

  const before = state.bytes();
  const staleContext = { hasRecentUnresolvedConfirmationPrompt: true };
  for (const command of [
    compound,
    "把买牛奶标为完成并将取快递删掉",
    "把买牛奶标为完成并设为高优先级",
  ]) {
    assert.equal(determineTurnToolPolicy(command, staleContext).mode, "read-only");
  }
  assert.match(await state.chat(compound), /请明确/);
  assert.equal(state.bytes(), before);
  const stale = fakeModel(using(listParcel, remove), saying("假称已删除。"));
  await state.chat(compound, { client: stale.client }, [user("删除准备面试"), pendingPrompt()]);
  assert.deepEqual(toolNames(stale.requests[0]), ["list_tasks"]);
  expectToolError(stale.requests[1], remove, "STALE_CONTEXT_READ_ONLY");
  assert.equal(state.bytes(), before);

  const recurring = call("update_tasks", { updates: [{ id: milk.id, priority: "high" }] });
  const deleted = call("update_tasks", { updates: [{ id: parcel.id, done: true }] });
  const normal = fakeModel(
    using(listMilk),
    using(listParcel),
    using(update, remove),
    using(recurring, deleted),
    saying("已更新牛奶并删除快递任务。"),
  );
  await state.chat(compound, { client: normal.client });
  for (const tool of [update, remove, recurring]) expectToolData(normal.requests[4], tool);
  expectToolError(normal.requests[4], deleted, "TASK_NOT_FOUND");
  assert.deepEqual(state.disk(), [{ ...milk, done: true, priority: "high" }]);

  const multiField = call("update_tasks", {
    updates: [{ id: milk.id, done: false, priority: "low" }],
  });
  const combined = fakeModel(using(listMilk, multiField), saying("已恢复待办并调整优先级。"));
  await state.chat("把买牛奶标记为未完成并设为低优先级", { client: combined.client });
  expectToolData(combined.requests[1], multiField);
  assert.deepEqual(state.disk(), [{ ...milk, done: false, priority: "low" }]);
});

test("R4 聊天确认工具载荷只含摘要，令牌留在服务端", async (t) => {
  const state = fixture(t);
  const entries = family(state);
  const [parent] = entries;
  const milk = state.tasks.create("买牛奶");
  const before = state.bytes();
  const duplicate = call("create_tasks", { tasks: [{ title: milk.title }] });
  const remove = call("delete_tasks", { ids: [parent.id] });
  const list = call("list_tasks", { query: parent.title });
  const cases = [
    { kind: "duplicate-create", command: "添加买牛奶", calls: [duplicate], tool: duplicate },
    { kind: "cascade-delete", command: "删除准备面试", calls: [list, remove], tool: remove },
  ] as const;
  for (const { kind, command, calls, tool } of cases) {
    const fake = fakeModel(using(...calls), saying("请确认操作。"));
    await state.chat(command, { conversationId: kind, client: fake.client });
    const result = expectToolError(fake.requests[1], tool, "CONFIRMATION_REQUIRED");
    const pending = state.pendingConfirmations.getByConversationId(kind);
    assert(pending && pending.kind === kind);
    const payload = fake.requests[1].messages.find(
      (message) => message.role === "tool" && message.tool_call_id === tool.id,
    );
    assert(payload?.role === "tool" && typeof payload.content === "string");
    assert(!payload.content.includes("confirmationId"));
    assert(!payload.content.includes(pending.confirmationId));
    assert.deepEqual(
      result.details,
      kind === "duplicate-create"
        ? {
            expiresAt: pending.expiresAt,
            duplicateTitles: [milk.title],
            duplicateTaskIds: [milk.id],
          }
        : { expiresAt: pending.expiresAt, affectedTasks: entries },
    );
    assert.equal(state.bytes(), before);
  }
});

test("R5 工具参数纠错后保留正常汇总，策略拦截提示仍有效", async (t) => {
  const state = fixture(t);
  const milk = state.tasks.create("买牛奶", { dueDate: "2026-02-28" });
  const before = state.bytes();
  const invalid = call("list_tasks", { dueDateFrom: "2026-02-30" });
  const valid = call("list_tasks", { dueDateFrom: milk.dueDate });
  const summary = "找到一条任务：买牛奶，2 月 28 日到期。";
  const fake = fakeModel(using(invalid), using(valid), saying(summary));
  const answer = await state.chat("我有哪些任务？", { client: fake.client });
  expectToolError(fake.requests[1], invalid, "INVALID_ARGUMENTS");
  assert.equal(expectToolData<ListedTasks>(fake.requests[2], valid).tasks[0].id, milk.id);
  assert.equal(answer, summary);

  const blocked = call("delete_tasks", { ids: [milk.id] });
  const rejecting = fakeModel(using(blocked), using(valid), saying("假称已删除。"));
  const staleAnswer = await state.chat("我有哪些任务？", { client: rejecting.client }, [
    user("删除过期任务"),
    pendingPrompt(),
  ]);
  expectToolError(rejecting.requests[1], blocked, "STALE_CONTEXT_READ_ONLY");
  expectToolData(rejecting.requests[2], valid);
  assert.match(staleAnswer, /不能执行历史待确认操作/);
  assert.doesNotMatch(staleAnswer, /已删除/);
  assert.equal(state.bytes(), before);
});

test("R6 scoped-write 允许匹配根任务的常规属性，仍拒绝其余目标", async (t) => {
  const state = fixture(t);
  const parent = state.tasks.create("其他任务");
  const before = state.bytes();
  const history = [user("删除准备面试"), pendingPrompt()];
  const command = "现在请重新添加写周报";
  const task = { title: "写周报", priority: "high", dueDate: "2026-12-04", effort: "medium" };
  const rejected = [
    call("create_tasks", { tasks: [{ ...task, title: "另一条任务" }] }),
    call("create_tasks", { tasks: [task, { title: "另一条任务" }] }),
    call("create_tasks", { tasks: [{ ...task, parentId: parent.id }] }),
    call("create_tasks", { tasks: [{ ...task, subtasks: [{ title: "额外子任务" }] }] }),
  ];
  const create = call("create_tasks", { tasks: [task] });
  const fake = fakeModel(
    using(...rejected),
    (request) => {
      for (const tool of rejected) expectToolError(request, tool, "CURRENT_TURN_INTENT_MISMATCH");
      assert.equal(state.bytes(), before);
      return using(create);
    },
    saying("已新增写周报。"),
  );
  await state.chat(command, { client: fake.client }, history);
  assert.deepEqual(toolNames(fake.requests[0]), ["list_tasks", "create_tasks"]);
  expectToolData(fake.requests[2], create);
  const [kept, added] = state.disk();
  assert(added.id);
  assert.deepEqual(state.disk(), [parent, { id: added.id, done: false, ...task }]);
  assert.deepEqual(kept, parent);
});

test("R6 单参数调用 reply 兼容原有入口", async () => {
  const answer = await reply([user("你好")]);
  assert.match(answer, /请明确说明操作和任务名称/);
});
