'use client';

import { useEffect, useState } from 'react';
import { setupWalletSelector } from '@near-wallet-selector/core';
import { setupModal } from '@near-wallet-selector/modal-ui';
import { setupMyNearWallet } from "@near-wallet-selector/my-near-wallet";
import { nearConfig } from '../lib/near-config';
import "@near-wallet-selector/modal-ui/styles.css";

export default function NearWallet() {
  const [accountId, setAccountId] = useState<string | null>(null);
  const [modal, setModal] = useState<any>(null);

  useEffect(() => {
    setupWalletSelector({
      network: "testnet",
      modules: [setupMyNearWallet()],
    }).then((selector) => {
      const modal = setupModal(selector, {
        contractId: nearConfig.contractName,
      });
      setModal(modal);

      // Get existing account
      selector.wallet().then((wallet) => {
        wallet.getAccounts().then((accounts) => {
          if (accounts.length > 0) {
            setAccountId(accounts[0].accountId);
          }
        });
      });
    });
  }, []);

  const handleSignIn = () => {
    modal.show();
  };

  const handleSignOut = async () => {
    const wallet = await modal.selector.wallet();
    wallet.signOut();
    setAccountId(null);
    window.location.reload();
  };

  return (
    <div className="flex items-center gap-4">
      {accountId ? (
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-300">
            {accountId}
          </span>
          <button
            onClick={handleSignOut}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg text-sm transition-colors"
          >
            Disconnect
          </button>
        </div>
      ) : (
        <button
          onClick={handleSignIn}
          className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors"
        >
          Connect NEAR Wallet
        </button>
      )}
    </div>
  );
} 