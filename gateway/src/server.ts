//server.ts
import { getEnvVariables } from "./config";
import { Server } from "http";
import { createApp } from "./app";
import { connectDB, disconnectDB } from "./database";
import { startDiscordBot, isBotReady } from "./services/discord/discordBotService";

let server: Server | null = null;

export const startServer = async (): Promise<Server> => {``
    const { MONGODB_URI, PORT, ALLOWED_ORIGIN } = getEnvVariables();
    await connectDB(MONGODB_URI);
    const app = createApp();

    await startDiscordBot();

    while (!isBotReady()) {
        await new Promise((resolve) => setTimeout(resolve, 100));
    }

    server = app.listen(PORT, () => {
        console.log(`Server is running at http://localhost:${PORT}`);
        console.log("ALLOWED ORIGINS: ", ALLOWED_ORIGIN);
    });
    return server;
};

export const stopServer = async (): Promise<void> => {
    if (server) {
        server.close();
        server = null;
    }
    await disconnectDB();
};

startServer();
