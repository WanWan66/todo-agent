import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { z } from "zod";

export { normalizePriority } from "../shared/task-display.ts";

const idSchema = z.string().min(1);
const titleSchema = z.string().trim().min(1).max(200);
const priorityLevelSchema = z.enum(["low", "medium", "high"]);
const taskFields = {
  dueDate: z.string().refine(isValidIsoDate).optional(),
  priority: z.union([priorityLevelSchema, z.number()]).optional(),
  effort: z.enum(["small", "medium", "large"]).optional(),
  parentId: idSchema.optional(),
};

const storedTaskSchema = z.looseObject({
  id: idSchema,
  title: z.string(),
  done: z.boolean(),
  ...taskFields,
});
const taskListSchema = z.array(storedTaskSchema);

const singleTaskInputSchema = z.strictObject({ title: titleSchema, ...taskFields });
const newSubtaskSchema = singleTaskInputSchema
  .omit({ parentId: true })
  .extend({ priority: priorityLevelSchema.optional() });
const newTaskSchema = newSubtaskSchema.extend({
  parentId: taskFields.parentId,
  subtasks: z.array(newSubtaskSchema).optional(),
});

const changesSchema = z.strictObject({
  title: titleSchema.optional(),
  done: z.boolean().optional(),
  dueDate: taskFields.dueDate.nullable(),
  priority: taskFields.priority.nullable(),
  effort: taskFields.effort.nullable(),
  parentId: taskFields.parentId.nullable(),
});
const updateSchema = changesSchema.extend({ id: idSchema });

export type Task = z.infer<typeof storedTaskSchema>;
export type NewSubtaskInput = z.infer<typeof newSubtaskSchema>;
export type NewTaskInput = NewSubtaskInput &
  ({ parentId: string; subtasks?: never } | { parentId?: undefined; subtasks?: NewSubtaskInput[] });
export type TaskChanges = z.infer<typeof changesSchema>;
export type TaskUpdate = z.infer<typeof updateSchema>;
type TaskFields = Omit<z.infer<typeof singleTaskInputSchema>, "title">;

export const MAX_BATCH_SIZE = 10;

type TaskStoreErrorCode = "TASK_NOT_FOUND" | "INVALID_PARENT_TASK" | "BATCH_LIMIT_EXCEEDED";

export class TaskStoreError extends Error {
  readonly code: TaskStoreErrorCode;

  constructor(code: TaskStoreErrorCode, message: string) {
    super(message);
    this.name = "TaskStoreError";
    this.code = code;
  }
}

export interface TaskStore {
  list(): Task[];
  create(title: string, fields?: TaskFields): Task;
  createMany(items: NewTaskInput[]): Task[];
  update(id: string, changes: TaskChanges): Task | undefined;
  updateMany(items: TaskUpdate[]): Task[];
  remove(id: string): boolean;
  removeMany(ids: string[]): Task[];
}

export function isValidIsoDate(value: unknown): boolean {
  if (typeof value !== "string" || value.length !== 10 || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    return false;
  }
  const date = new Date(value + "T00:00:00.000Z");
  return !Number.isNaN(date.getTime()) && date.toISOString().slice(0, 10) === value;
}

export function getDeletionTargets(tasks: Task[], ids: string[]): Task[] | undefined {
  const selected = new Set(ids);
  const existing = new Set(tasks.map((task) => task.id));
  if (ids.some((id) => !existing.has(id))) return undefined;

  return tasks.filter(
    (task) => selected.has(task.id) || (task.parentId !== undefined && selected.has(task.parentId)),
  );
}

function validateHierarchy(tasks: Task[]) {
  const byId = new Map(tasks.map((task) => [task.id, task]));
  for (const task of tasks) {
    if (task.parentId === undefined) continue;
    const parent = byId.get(task.parentId);
    if (!parent || parent.id === task.id || parent.parentId !== undefined) {
      throw new TaskStoreError("INVALID_PARENT_TASK", "父任务必须是另一条已存在的根任务。");
    }
  }
}

function buildTask(title: string, { dueDate, priority, effort, parentId }: TaskFields = {}): Task {
  const task: Task = { id: randomUUID(), title, done: false };
  if (dueDate !== undefined) task.dueDate = dueDate;
  if (priority !== undefined) task.priority = priority;
  if (effort !== undefined) task.effort = effort;
  if (parentId !== undefined) task.parentId = parentId;
  return task;
}

function applyChanges(task: Task, changes: TaskChanges) {
  for (const [key, value] of Object.entries(changes)) {
    if (value === undefined) continue;
    if (value === null) delete task[key];
    else task[key] = value;
  }
}

export function createJsonTaskStore(filename: string): TaskStore {
  function list(): Task[] {
    if (!existsSync(filename)) return [];
    return taskListSchema.parse(JSON.parse(readFileSync(filename, "utf8")));
  }

  // 同一服务进程内同步完成读写；替换文件前，原清单始终完整可读。
  function save(tasks: Task[]) {
    validateHierarchy(tasks);
    mkdirSync(dirname(filename), { recursive: true });
    const temporary = filename + ".tmp";
    writeFileSync(temporary, JSON.stringify(tasks, null, 2) + "\n");
    renameSync(temporary, filename);
  }

  return {
    list,

    create(title, fields) {
      const input = singleTaskInputSchema.parse({ ...fields, title });
      const task = buildTask(input.title, input);
      save([...list(), task]);
      return task;
    },

    createMany(items) {
      const inputs = z.array(newTaskSchema).min(1).parse(items);
      const count = inputs.reduce((total, item) => total + 1 + (item.subtasks?.length ?? 0), 0);
      if (count > MAX_BATCH_SIZE) {
        throw new TaskStoreError("BATCH_LIMIT_EXCEEDED", "每批最多创建 10 条任务（含子任务）。");
      }

      const created: Task[] = [];
      for (const input of inputs) {
        if (input.parentId !== undefined && input.subtasks?.length) {
          throw new TaskStoreError("INVALID_PARENT_TASK", "只有根任务可以附带子任务。");
        }
        const task = buildTask(input.title, input);
        created.push(task);
        for (const subtask of input.subtasks ?? []) {
          created.push(buildTask(subtask.title, { ...subtask, parentId: task.id }));
        }
      }

      save([...list(), ...created]);
      return created;
    },

    update(id, changes) {
      const tasks = list();
      const task = tasks.find((item) => item.id === id);
      if (!task) return undefined;

      applyChanges(task, changesSchema.parse(changes));
      save(tasks);
      return task;
    },

    updateMany(items) {
      const inputs = z.array(updateSchema).min(1).parse(items);
      if (inputs.length > MAX_BATCH_SIZE) {
        throw new TaskStoreError("BATCH_LIMIT_EXCEEDED", "每批最多更新 10 条任务。");
      }

      const tasks = list();
      const updated = inputs.map(({ id, ...changes }) => {
        const task = tasks.find((item) => item.id === id);
        if (!task) throw new TaskStoreError("TASK_NOT_FOUND", "找不到这条任务。");
        applyChanges(task, changes);
        return task;
      });
      save(tasks);
      return updated;
    },

    remove(id) {
      const tasks = list();
      const remaining = tasks.filter((task) => task.id !== id);
      if (remaining.length === tasks.length) return false;

      save(remaining);
      return true;
    },

    removeMany(ids) {
      const selected = new Set(z.array(idSchema).min(1).parse(ids));
      const tasks = list();
      const removed = tasks.filter((task) => selected.has(task.id));
      if (new Set(removed.map((task) => task.id)).size !== selected.size) {
        throw new TaskStoreError("TASK_NOT_FOUND", "找不到这条任务。");
      }
      save(tasks.filter((task) => !selected.has(task.id)));
      return removed;
    },
  };
}
