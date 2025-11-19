"use client";

import Section from "@/components/common/Section";
import TradesTable from "@/components/common/TradesTable";
import { useTrades } from "@/hooks/useBackend";

export default function TradesPage() {
  const { data, isLoading } = useTrades(200);

  return (
    <Section title="Trade History">
      <TradesTable trades={data ?? []} isLoading={isLoading} />
    </Section>
  );
}

