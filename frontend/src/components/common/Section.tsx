"use client";

import { ReactNode } from "react";

interface SectionProps {
  title: string;
  description?: string;
  children: ReactNode;
}

export default function Section({
  title,
  description,
  children
}: SectionProps) {
  return (
    <section className="space-y-3">
      <div>
        <h2 className="text-lg font-semibold">{title}</h2>
        {description && (
          <p className="text-sm text-slate-400">{description}</p>
        )}
      </div>
      <div className="bg-card rounded-lg p-4">{children}</div>
    </section>
  );
}

