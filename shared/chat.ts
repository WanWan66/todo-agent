import * as z from "zod/mini";

export const chatMessageMetadataSchema = z.strictObject({
  kind: z.optional(z.enum(["pending_confirmation_prompt", "confirmation_resolved"])),
  confirmationKind: z.optional(z.enum(["duplicate-create", "cascade-delete"])),
});

export type ChatMessageMetadata = z.infer<typeof chatMessageMetadataSchema>;

const messageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string().check(z.trim(), z.minLength(1)),
  metadata: z.optional(chatMessageMetadataSchema),
});

export type ChatMessage = z.infer<typeof messageSchema>;

export const chatRequestSchema = z.object({
  messages: z.array(messageSchema).check(
    z.minLength(1),
    z.refine((messages) => messages.at(-1)?.role === "user"),
  ),
  conversationId: z.optional(z.string().check(z.trim(), z.minLength(1))),
});
