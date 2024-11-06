import { useState, useEffect } from 'react';
import { connect, WalletConnection } from 'near-api-js';
import { nearConfig } from '../lib/near-config';

export function useWallet() {
  const [wallet, setWallet] = useState<WalletConnection | null>(null);
  const [accountId, setAccountId] = useState<string | null>(null);

  useEffect(() => {
    const initNear = async () => {
      const near = await connect(nearConfig);
      const wallet = new WalletConnection(near, 'federated-learning');
      setWallet(wallet);
      
      if (wallet.isSignedIn()) {
        setAccountId(wallet.getAccountId());
      }
    };

    initNear();
  }, []);

  return { wallet, accountId };
} 