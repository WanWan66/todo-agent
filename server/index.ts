import { join } from "node:path";
import { reply } from "./agent.ts";
import { createApp } from "./app.ts";
import { createPendingConfirmationStore } from "./pending-confirmations.ts";
import { createJsonTaskStore } from "./task-store.ts";

const tasks = createJsonTaskStore(join("data", "tasks.json"));
const pendingConfirmations = createPendingConfirmationStore();
const app = createApp({ reply, tasks, pendingConfirmations });
const port = Number(process.env.PORT || 3000);
app.listen(port, "127.0.0.1", () => console.log(`小明智能体已启动：http://127.0.0.1:${port}`));
