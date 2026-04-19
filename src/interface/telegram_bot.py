"""Telegram bot — spec §4.6. python-telegram-bot v20+ asyncio native.

Commands: /positions, /status, /halt, /resume, /analyze [symbol]
Free-text → ChatLLM.converse()
stop_signals=None avoids signal handler collision with APScheduler.
"""
from __future__ import annotations

from typing import Any

import structlog
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

log = structlog.get_logger()


def parse_analyze_command(text: str) -> str:
    """Extract symbol from /analyze command. Default ETHUSDT."""
    parts = text.strip().split()
    if len(parts) >= 2:
        return parts[1].upper()
    return "ETHUSDT"


class TelegramBot:
    """Wraps python-telegram-bot Application for integration with Orchestrator TaskGroup."""

    def __init__(
        self,
        token: str,
        chat_llm: Any = None,       # ChatLLM — injected
        halt_manager: Any = None,    # HaltManager — injected
        scan_fn: Any = None,         # on_demand scan callback
        broker: Any = None,          # Broker — for /positions
        engine: Any = None,          # sa.Engine — for /status queries
    ) -> None:
        self._token = token
        self._chat_llm = chat_llm
        self._halt = halt_manager
        self._scan_fn = scan_fn
        self._broker = broker
        self._engine = engine
        self._app: Application | None = None

    def build(self) -> Application:
        """Build the Application with handlers. Call before start()."""
        self._app = (
            Application.builder()
            .token(self._token)
            .build()
        )
        self._app.add_handler(CommandHandler("positions", self._cmd_positions))
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("halt", self._cmd_halt))
        self._app.add_handler(CommandHandler("resume", self._cmd_resume))
        self._app.add_handler(CommandHandler("analyze", self._cmd_analyze))
        self._app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._free_text))
        return self._app

    async def start(self) -> None:
        """Initialize and start polling. Use stop_signals=None for TaskGroup compat."""
        if self._app is None:
            self.build()
        assert self._app is not None
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(stop_signals=None)  # type: ignore[union-attr]
        log.info("telegram_bot_started")

    async def stop(self) -> None:
        if self._app is not None:
            await self._app.updater.stop()  # type: ignore[union-attr]
            await self._app.stop()
            await self._app.shutdown()
            log.info("telegram_bot_stopped")

    async def send_message(self, chat_id: int | str, text: str) -> None:
        """Send a message (used by scan pipeline for notifications)."""
        if self._app and self._app.bot:
            await self._app.bot.send_message(chat_id=chat_id, text=text)

    # ── Command handlers ───────────────────────────────────────────

    async def _cmd_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._broker or not update.effective_message:
            return
        positions = await self._broker.positions()
        if not positions:
            await update.effective_message.reply_text("No open positions.")
            return
        lines = [f"• {p.symbol}: {p.qty:+.4f} @ {p.avg_entry:.2f}" for p in positions]
        await update.effective_message.reply_text("\n".join(lines))

    async def _cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message:
            return
        halted = self._halt.is_halted() if self._halt else False
        status = "🔴 HALTED" if halted else "🟢 Running"
        await update.effective_message.reply_text(f"Status: {status}")

    async def _cmd_halt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._halt or not update.effective_message:
            return
        self._halt.activate(source="telegram", reason="Manual /halt command")
        await update.effective_message.reply_text("HALT activated via Telegram.")

    async def _cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._halt or not update.effective_message:
            return
        ok, still_active = self._halt.attempt_resume()
        if ok:
            await update.effective_message.reply_text("Resumed successfully.")
        else:
            await update.effective_message.reply_text(
                f"Cannot resume — triggers still breached: {', '.join(still_active)}"
            )

    async def _cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_message or not self._scan_fn:
            return
        symbol = parse_analyze_command(update.effective_message.text or "/analyze")
        await update.effective_message.reply_text(f"Analyzing {symbol}...")
        try:
            result = await self._scan_fn(symbol)
            await update.effective_message.reply_text(result or "Analysis complete — no signal.")
        except Exception as e:
            log.exception("analyze_failed", symbol=symbol)
            await update.effective_message.reply_text(f"Analysis failed: {e}")

    async def _free_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._chat_llm or not update.effective_message:
            return
        chat_id = str(update.effective_chat.id) if update.effective_chat else "unknown"
        user_text = update.effective_message.text or ""

        from interface.repositories import ConversationRepo
        conv_repo = ConversationRepo(self._engine)
        cid = conv_repo.get_by_chat_id(chat_id)
        if not cid:
            cid = conv_repo.create(chat_id)

        reply = await self._chat_llm.converse(cid, user_text)
        await update.effective_message.reply_text(reply or "I couldn't generate a response.")
