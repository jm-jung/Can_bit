import {
  useMutation,
  useQuery,
  useQueryClient
} from "@tanstack/react-query";
import apiClient from "@/lib/apiClient";
import {
  DailyReport,
  EquityPoint,
  LogEntry,
  MonitorResponse,
  RealtimeResponse,
  RiskStatus
} from "@/lib/types";

const fetcher = <T>(url: string) =>
  apiClient.get<T>(url).then((res) => res.data);

export const useRealtimeLast = () =>
  useQuery({
    queryKey: ["realtime-last"],
    queryFn: () => fetcher<RealtimeResponse>("/realtime/last"),
    refetchInterval: 10_000
  });

export const useMonitor = () =>
  useQuery({
    queryKey: ["monitor"],
    queryFn: () => fetcher<MonitorResponse>("/backoffice/monitor"),
    refetchInterval: 10_000
  });

export const useTrades = (limit = 100) =>
  useQuery({
    queryKey: ["logs", "trades", limit],
    queryFn: () => fetcher<LogEntry[]>(`/backoffice/logs/trades?limit=${limit}`),
    refetchInterval: 30_000
  });

export const useRiskLogs = (limit = 100) =>
  useQuery({
    queryKey: ["logs", "risk", limit],
    queryFn: () => fetcher<LogEntry[]>(`/backoffice/logs/risk?limit=${limit}`),
    refetchInterval: 30_000
  });

export const useErrorLogs = (limit = 100) =>
  useQuery({
    queryKey: ["logs", "errors", limit],
    queryFn: () => fetcher<LogEntry[]>(`/backoffice/logs/errors?limit=${limit}`),
    refetchInterval: 30_000
  });

export const useDailyReport = () =>
  useQuery({
    queryKey: ["daily-report"],
    queryFn: () => fetcher<DailyReport>("/backoffice/daily-report"),
    refetchInterval: 60_000
  });

export const useEquityCurve = () =>
  useQuery({
    queryKey: ["equity-curve"],
    queryFn: () => fetcher<EquityPoint[]>("/backoffice/equity-curve"),
    refetchInterval: 60_000
  });

export const useRiskStatus = () =>
  useQuery({
    queryKey: ["risk-status"],
    queryFn: () => fetcher<RiskStatus>("/risk/status"),
    refetchInterval: 15_000
  });

export const useTradeMode = () =>
  useQuery({
    queryKey: ["trade-mode"],
    queryFn: () => fetcher<{ mode: "SIM" | "REAL" }>("/trade/mode"),
    refetchInterval: 15_000
  });

export const useLiveMode = () =>
  useQuery({
    queryKey: ["live-mode"],
    queryFn: () =>
      fetcher<{ live_mode: boolean; warning?: string }>("/trade/live-mode"),
    refetchInterval: 15_000
  });

export const useSetTradeMode = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (mode: "SIM" | "REAL") =>
      apiClient.post(`/trade/mode/${mode}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trade-mode"] });
      qc.invalidateQueries({ queryKey: ["monitor"] });
    }
  });
};

export const useSetLiveMode = () => {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (flag: "on" | "off") =>
      apiClient.post(`/trade/live-mode/${flag}`),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["live-mode"] });
      qc.invalidateQueries({ queryKey: ["monitor"] });
    }
  });
};

