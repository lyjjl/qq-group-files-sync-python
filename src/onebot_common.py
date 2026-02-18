from __future__ import annotations


def require_ok(action: str, result) -> None:
    rc = getattr(result, "retcode", None)
    st = getattr(result, "status", None)
    if rc != 0 or (st not in {"ok", "OK", "", None}):
        msg = getattr(result, "message", None)
        wording = getattr(result, "wording", None)
        raise RuntimeError(
            "OneBot API failed: "
            f"{action} retcode={rc} status={st} message={msg!r} wording={wording!r}"
        )
