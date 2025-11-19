"use client";

import { useState } from "react";
import { useLiveMode, useSetLiveMode } from "@/hooks/useBackend";
import ConfirmDialog from "@/components/common/ConfirmDialog";

export default function LiveModeToggle() {
  const { data, isLoading } = useLiveMode();
  const [open, setOpen] = useState(false);
  const mutation = useSetLiveMode();
  const isLive = data?.live_mode ?? false;

  const toggle = () => setOpen(true);
  const handleConfirm = () => {
    mutation.mutate(isLive ? "off" : "on", {
      onSettled: () => setOpen(false)
    });
  };

  return (
    <div className="space-y-2">
      <p className="text-sm text-slate-400">Live Trading</p>
      <button
        disabled={isLoading || mutation.isPending}
        onClick={toggle}
        className={`w-full rounded px-3 py-2 text-sm font-semibold ${
          isLive ? "bg-danger hover:bg-danger/90" : "bg-slate-600 hover:bg-slate-500"
        }`}
      >
        {isLive ? "Turn Live OFF" : "Turn Live ON"}
      </button>
      <ConfirmDialog
        title="Confirm Live Trading"
        description="Live 모드를 활성화하면 실제 주문이 시도될 수 있습니다. 정말로 진행하시겠습니까?"
        open={open}
        onCancel={() => setOpen(false)}
        onConfirm={handleConfirm}
        confirmLabel={isLive ? "Disable" : "Enable"}
        confirmVariant={isLive ? "danger" : "warning"}
      />
    </div>
  );
}

