import { Router, type ErrorRequestHandler, type Response } from "express";
import { z } from "zod";
import { dateInTimeZone, DEFAULT_APP_TIME_ZONE } from "../shared/task-display.ts";
import {
  createPendingConfirmationStore,
  type ConfirmationScope,
  type ConsumeConfirmationResult,
  type PendingConfirmationStore,
} from "./pending-confirmations.ts";
import {
  getDeletionTargets,
  isValidIsoDate,
  TaskStoreError,
  type TaskStore,
} from "./task-store.ts";

const titleSchema = z.string().trim().min(1).max(200);
const optionalFields = {
  dueDate: z.string().refine(isValidIsoDate).optional(),
  priority: z.union([z.enum(["low", "medium", "high"]), z.number()]).optional(),
  effort: z.enum(["small", "medium", "large"]).optional(),
  parentId: z.string().min(1).optional(),
};
const createTaskSchema = z.strictObject({ title: titleSchema, ...optionalFields });
const updateTaskSchema = z
  .strictObject({
    title: titleSchema.optional(),
    done: z.boolean().optional(),
    dueDate: optionalFields.dueDate.nullable(),
    priority: optionalFields.priority.nullable(),
    effort: optionalFields.effort.nullable(),
    parentId: optionalFields.parentId.nullable(),
  })
  .refine((changes) => Object.values(changes).some((value) => value !== undefined));

const taskNotFound = { error: "找不到这条任务。" };
const restScope: ConfirmationScope = { channel: "rest" };

function sendConfirmationFailure(
  res: Response,
  reason: Extract<ConsumeConfirmationResult, { ok: false }>["reason"],
  expiredMessage: string,
) {
  if (reason === "EXPIRED") {
    return res.status(409).json({ error: "CONFIRMATION_EXPIRED", message: expiredMessage });
  }
  return res
    .status(404)
    .json({ error: "CONFIRMATION_NOT_FOUND", message: "找不到有效的待确认删除操作。" });
}

const handleTaskError: ErrorRequestHandler = (error, _req, res, next) => {
  if (error instanceof TaskStoreError && error.code === "INVALID_PARENT_TASK") {
    return res.status(400).json({ error: error.message });
  }
  next(error);
};

export function taskRouter(
  tasks: TaskStore,
  pendingConfirmations: PendingConfirmationStore = createPendingConfirmationStore(),
) {
  const router = Router();

  router.get("/", (_req, res) => {
    const now = new Date();
    try {
      res.set(
        "X-Task-Today",
        dateInTimeZone(now, process.env.APP_TIME_ZONE || DEFAULT_APP_TIME_ZONE),
      );
    } catch (error) {
      if (!(error instanceof RangeError)) throw error;
      res.set("X-Task-Today", dateInTimeZone(now));
    }
    res.json(tasks.list());
  });

  router.post("/", (req, res) => {
    const input = createTaskSchema.safeParse(req.body);
    if (!input.success) {
      const error = input.error.issues.some((issue) => issue.path[0] === "title")
        ? "请提供 1 至 200 个字符的任务标题。"
        : "请提供有效的任务字段。";
      return res.status(400).json({ error });
    }
    const { title, ...fields } = input.data;
    res.status(201).json(tasks.create(title, fields));
  });

  router.patch("/:id", (req, res) => {
    const input = updateTaskSchema.safeParse(req.body);
    if (!input.success) {
      return res.status(400).json({ error: "请至少提供一个有效的可更新字段。" });
    }

    const updated = tasks.update(req.params.id, input.data);
    if (!updated) return res.status(404).json(taskNotFound);
    res.json(updated);
  });

  router.delete("/:id", (req, res) => {
    const { id } = req.params;
    const affectedTasks = getDeletionTargets(tasks.list(), [id]);
    if (!affectedTasks) return res.status(404).json(taskNotFound);
    if (affectedTasks.length === 1) {
      tasks.remove(id);
      return res.sendStatus(204);
    }

    const { confirmationId } = pendingConfirmations.create({
      kind: "cascade-delete",
      channel: "rest",
      rootTaskIds: [id],
      affectedTaskIds: affectedTasks.map((task) => task.id),
    });
    return res.status(409).json({
      error: "CONFIRMATION_REQUIRED",
      confirmationId,
      affectedTasks,
      message: "删除这条任务也会删除它的子任务，请确认。",
    });
  });

  router.post("/confirmations/:confirmationId", (req, res) => {
    const result = pendingConfirmations.consume(req.params.confirmationId, restScope);
    if (!result.ok) {
      return sendConfirmationFailure(res, result.reason, "确认已过期，请重新发起删除。");
    }

    const { confirmation } = result;
    // REST 归属校验只允许级联删除记录进入此分支。
    if (confirmation.channel !== "rest") {
      return sendConfirmationFailure(res, "FORBIDDEN", "确认已过期，请重新发起删除。");
    }

    const { rootTaskIds, affectedTaskIds } = confirmation;
    const currentTargets = getDeletionTargets(tasks.list(), rootTaskIds);
    const snapshotIds = new Set(affectedTaskIds);
    if (
      !currentTargets ||
      currentTargets.length !== snapshotIds.size ||
      currentTargets.some((task) => !snapshotIds.has(task.id))
    ) {
      return res.status(409).json({
        error: "CONFIRMATION_TARGET_CHANGED",
        message: "任务或子任务已变更，请重新发起删除。",
      });
    }

    tasks.removeMany(affectedTaskIds);
    res.sendStatus(204);
  });

  router.delete("/confirmations/:confirmationId", (req, res) => {
    const result = pendingConfirmations.cancel(req.params.confirmationId, restScope);
    if (!result.ok) return sendConfirmationFailure(res, result.reason, "确认已过期。");
    res.sendStatus(204);
  });

  router.use(handleTaskError);
  return router;
}
