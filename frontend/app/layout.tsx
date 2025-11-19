import type { Metadata } from "next";
import "./globals.css";
import Providers from "./providers";
import NavBar from "@/components/layout/NavBar";

export const metadata: Metadata = {
  title: "BTC Trading Dashboard",
  description: "Monitor FastAPI Bitcoin trading backend"
};

export default function RootLayout({
  children
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="ko">
      <body>
        <Providers>
          <div className="min-h-screen bg-background">
            <NavBar />
            <main className="px-4 py-6 md:px-10">{children}</main>
          </div>
        </Providers>
      </body>
    </html>
  );
}

