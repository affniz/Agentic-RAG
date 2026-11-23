import { useState, type FormEvent } from "react";
import { api } from "./api";
import { Icon } from "./components";

export function LoginPage({ onLogin, expired }: { onLogin: () => void; expired: boolean }) {
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(expired ? "Your session ended. Please sign in again." : null);

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await api.login(password);
      onLogin();
    } catch (err) {
      setError((err as Error).message === "wrong password" ? "Wrong password." : (err as Error).message);
      setPassword("");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="grid h-full place-items-center px-4">
      <form onSubmit={submit} className="w-full max-w-sm rounded-2xl border border-zinc-200 p-6 shadow-sm dark:border-zinc-800">
        <div className="mb-5 flex items-center gap-2">
          <div className="grid h-7 w-7 place-items-center rounded-md bg-indigo-600 text-white">{Icon.search("h-4 w-4")}</div>
          <h1 className="font-semibold tracking-tight">Agentic RAG</h1>
        </div>
        <label htmlFor="password" className="text-sm font-medium">Password</label>
        <input id="password" type="password" autoComplete="current-password" autoFocus required value={password}
          onChange={(e) => setPassword(e.target.value)}
          className="mt-1.5 w-full rounded-lg border border-zinc-200 bg-transparent px-3 py-2 text-sm dark:border-zinc-800 dark:bg-zinc-950" />
        {error && <p role="alert" className="mt-3 text-sm text-red-600 dark:text-red-400">{error}</p>}
        <button type="submit" disabled={busy || !password}
          className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50">
          {busy && Icon.spinner()} Sign in
        </button>
      </form>
    </main>
  );
}
