'use client';

import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { setupWalletSelector } from '@near-wallet-selector/core';
import { setupModal } from '@near-wallet-selector/modal-ui';
import { setupNearWallet } from "@near-wallet-selector/near-wallet";
import { setupMyNearWallet } from "@near-wallet-selector/my-near-wallet";
import { nearConfig } from '../lib/near-config';

interface WalletContextType {
  accountId: string | null;
  modal: any;
  selector: any;
}

const WalletContext = createContext<WalletContextType>({
  accountId: null,
  modal: null,
  selector: null,
});

export function WalletProvider({ children }: { children: ReactNode }) {
  const [accountId, setAccountId] = useState<string | null>(null);
  const [modal, setModal] = useState<any>(null);
  const [selector, setSelector] = useState<any>(null);

  useEffect(() => {
    setupWalletSelector({
      network: "testnet",
      modules: [setupNearWallet(), setupMyNearWallet()],
    }).then((selector) => {
      const modal = setupModal(selector, {
        contractId: nearConfig.contractName,
      });
      setModal(modal);
      setSelector(selector);

      // Get existing account
      selector.wallet().then((wallet: any) => {
        wallet.getAccounts().then((accounts: any[]) => {
          if (accounts.length > 0) {
            setAccountId(accounts[0].accountId);
          }
        });
      });
    });
  }, []);

  return (
    <WalletContext.Provider value={{ accountId, modal, selector }}>
      {children}
    </WalletContext.Provider>
  );
}

export function useWallet() {
  return useContext(WalletContext);
} 