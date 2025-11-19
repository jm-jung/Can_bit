"use client";

interface ConfirmDialogProps {
  title: string;
  description: string;
  open: boolean;
  onConfirm: () => void;
  onCancel: () => void;
  confirmLabel?: string;
  confirmVariant?: "danger" | "warning";
}

export default function ConfirmDialog({
  title,
  description,
  open,
  onConfirm,
  onCancel,
  confirmLabel = "Confirm",
  confirmVariant = "warning"
}: ConfirmDialogProps) {
  if (!open) return null;

  const confirmClass =
    confirmVariant === "danger"
      ? "bg-danger hover:bg-danger/90"
      : "bg-warning text-black hover:bg-warning/90";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="w-full max-w-md rounded-lg bg-card p-6 shadow-xl">
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="mt-2 text-sm text-slate-300">{description}</p>
        <div className="mt-6 flex justify-end gap-3 text-sm">
          <button
            onClick={onCancel}
            className="rounded border border-slate-500 px-4 py-2 hover:bg-slate-700"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className={`rounded px-4 py-2 font-semibold transition ${confirmClass}`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}

