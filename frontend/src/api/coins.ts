import { apiRequest } from "./client";

export type CoinTransaction = {
  id: string;
  amount: number;
  reason: string;
  report_id: string | null;
  note: string | null;
  created_at: string | null;
};

export type CoinWallet = {
  balance: number;
  total_earned: number;
  transactions: CoinTransaction[];
};

/*
 * Green-coin wallet of the logged-in citizen.
 *
 * Backend response:
 * { balance, total_earned, transactions: [...] }
 */
export async function getWallet(): Promise<CoinWallet> {
  return apiRequest<CoinWallet>(
    "/coins/me",
    {
      method: "GET",
    }
  );
}
