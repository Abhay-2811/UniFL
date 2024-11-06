'use client';

import { Card, Title, BarChart, Subtitle, Badge } from "@tremor/react";
import { useEffect, useState } from "react";
import axios from "axios";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title as ChartTitle,
  Tooltip,
  Legend,
  ChartData,
  ChartOptions
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import NearWallet from '../components/NearWallet';
import { useWallet } from '../contexts/WalletContext';
import { config } from '../lib/config';
import { providers } from "near-api-js";
import { nearConfig } from "@/lib/near-config";

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  ChartTitle,
  Tooltip,
  Legend
);

interface ModelStats {
  accuracy: number;
  loss: number;
  timestamp: string;
}

interface Contributor {
  client_id: string;
  contribution: number;
  total_impact: number;
  average_impact: number;
  last_update: string;
}

interface ServerLog {
  timestamp: string;
  message: string;
  type: 'info' | 'success' | 'warning' | 'error';
}

interface TokenDistribution {
  address: string;
  tokens: number;
  lastUpdate: string;
}

const THIRTY_TGAS = '30000000000000';
const NO_DEPOSIT = '0';

export default function Home() {
  const [modelStats, setModelStats] = useState<ModelStats[]>([]);
  const [contributors, setContributors] = useState<Contributor[]>([]);
  const [logs, setLogs] = useState<ServerLog[]>([]);
  const [tokenDistributions, setTokenDistributions] = useState<TokenDistribution[]>([]);
  const [nextDistribution, setNextDistribution] = useState<string>("");
  const { accountId, selector } = useWallet();
  const [userBalance, setUserBalance] = useState<string>('0');
  const [nextDistTime, setNextDistTime] = useState<string>('');
  const provider = new providers.JsonRpcProvider({url: nearConfig.nodeUrl});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsRes, contributorsRes, logsRes] = await Promise.all([
          axios.get('/api/model-stats'),
          axios.get('/api/contributors'),
          axios.get('/api/logs')
        ]);
        
        if (Array.isArray(statsRes.data)) setModelStats(statsRes.data);
        if (Array.isArray(contributorsRes.data)) setContributors(contributorsRes.data);
        if (Array.isArray(logsRes.data)) setLogs(logsRes.data);
        
        // Placeholder for token distribution data
        setTokenDistributions([
          { address: "0x742d...44e", tokens: 1500, lastUpdate: "2024-03-10" },
          { address: "0x123d...89f", tokens: 2300, lastUpdate: "2024-03-10" },
        ]);
        setNextDistribution("2024-03-17 00:00:00");
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const callMethod = async ({ contractId, method, args = {} }: { contractId: string, method: string, args?: any } ) => {
    if (!selector) return null;
    
    const wallet = await selector.wallet();
    const outcome = await wallet.signAndSendTransaction({
      receiverId: contractId,
      actions: [
        {
          type: 'FunctionCall',
          params: {
            methodName: method,
            args,
            gas: THIRTY_TGAS,
            deposit: NO_DEPOSIT,
          },
        },
      ],
    });
    
    return providers.getTransactionLastResult(outcome);
  };

  useEffect(() => {
    const fetchNearData = async () => {
      if (selector && accountId) {
        try {
          // Get user balance
          const balance = await callMethod({
            contractId: config.nearContractId,
            method: 'get_balance',
            args: { account_id: accountId }
          });
          console.log(balance);
          setUserBalance(balance as string);

          // Get next distribution time
          const nextDist = await callMethod({
            contractId: config.nearContractId,
            method: 'get_next_distribution_time',
            args: {}
          });
          setNextDistTime(new Date(Number(nextDist) / 1_000_000).toLocaleString());
        } catch (error) {
          console.error('Error fetching NEAR data:', error);
        }
      }
    };

    fetchNearData();
  }, [selector, accountId]);

  const chartData: ChartData<'line'> = {
    labels: modelStats.map(stat => {
      const date = new Date(stat.timestamp);
      return date.toLocaleTimeString();
    }),
    datasets: [
      {
        label: 'Accuracy',
        data: modelStats.map(stat => stat.accuracy),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        tension: 0.1,
        fill: true,
      },
      {
        label: 'Loss',
        data: modelStats.map(stat => stat.loss),
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        tension: 0.1,
        fill: true,
      },
    ],
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: 'Global Model Performance',
      },
    },
    scales: {
      y: {
        type: 'linear' as const,
        display: true,
        position: 'left' as const,
        beginAtZero: true,
        ticks: {
          callback: (value: number | string) => {
            if (typeof value === 'number') {
              return value.toFixed(3);
            }
            return value;
          }
        }
      },
      x: {
        type: 'category' as const,
        ticks: {
          maxRotation: 45,
          minRotation: 45
        }
      }
    }
  };

  return (
    <div className="min-h-screen p-8 bg-black text-white">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8 flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-white">{config.appName} Dashboard</h1>
            <p className="text-gray-400">Real-time model performance and contributor statistics</p>
          </div>
          <NearWallet />
        </div>

        {/* Token Balance Card - Show when wallet is connected */}
        {accountId && (
          <div className="mb-6">
            <Card className="p-6 bg-gray-800 border-gray-700">
              <div className="flex justify-between items-center">
                <div>
                  <Title className="text-white">Your Balance</Title>
                  <div className="text-2xl font-mono text-green-400 mt-2">
                    {(Number(userBalance) / 1e24).toFixed(2)} FLT
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-gray-400">Next Distribution</div>
                  <div className="text-sm text-gray-300 mt-1">{nextDistTime}</div>
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* Rest of the dashboard components */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Model Performance Chart */}
          <Card className="p-6 bg-gray-800 border-gray-700">
            <Title className="text-white">Model Performance</Title>
            <div className="h-[300px] mt-4">
              {modelStats.length > 0 ? (
                <Line 
                  data={{
                    ...chartData,
                    datasets: chartData.datasets.map(dataset => ({
                      ...dataset,
                      borderColor: dataset.label === 'Accuracy' ? '#4ade80' : '#f87171',
                      backgroundColor: dataset.label === 'Accuracy' ? 'rgba(74, 222, 128, 0.2)' : 'rgba(248, 113, 113, 0.2)',
                    }))
                  }} 
                  options={{
                    ...chartOptions,
                    plugins: {
                      ...chartOptions.plugins,
                      legend: {
                        ...chartOptions.plugins?.legend,
                        labels: {
                          color: 'white'
                        }
                      }
                    },
                    scales: {
                      y: {
                        ...chartOptions.scales?.y,
                        ticks: {
                          color: 'white',
                          callback: (value) => typeof value === 'number' ? value.toFixed(3) : value
                        },
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)'
                        }
                      },
                      x: {
                        ...chartOptions.scales?.x,
                        ticks: {
                          color: 'white',
                          maxRotation: 45,
                          minRotation: 45
                        },
                        grid: {
                          color: 'rgba(255, 255, 255, 0.1)'
                        }
                      }
                    }
                  }}
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <p className="text-gray-400">Waiting for model statistics...</p>
                </div>
              )}
            </div>
          </Card>

          {/* Server Logs */}
          <Card className="p-6 bg-gray-800 border-gray-700">
            <Title className="text-white mb-2">Server Logs</Title>
            <div className="h-[300px] overflow-y-auto mt-4 bg-black/50 rounded-lg p-4 font-mono text-sm scrollbar-thin scrollbar-thumb-gray-600 scrollbar-track-transparent">
              {logs.map((log, index) => (
                <div key={index} className="mb-1.5 flex">
                  <span className={`
                    mr-2
                    ${log.type === 'success' ? 'text-green-400' : 
                      log.type === 'warning' ? 'text-yellow-400' : 
                      log.type === 'error' ? 'text-red-400' : 
                      'text-blue-400'}
                  `}>
                    {log.type === 'success' ? '✓' :
                     log.type === 'warning' ? '⚠' :
                     log.type === 'error' ? '✗' :
                     '>'} 
                  </span>
                  <span className="text-gray-500">[{new Date(log.timestamp).toLocaleTimeString()}]</span>
                  <span className={`
                    ml-2
                    ${log.type === 'success' ? 'text-green-300' : 
                      log.type === 'warning' ? 'text-yellow-300' : 
                      log.type === 'error' ? 'text-red-300' : 
                      'text-gray-300'}
                  `}>
                    {log.message}
                  </span>
                </div>
              ))}
              {logs.length === 0 && (
                <div className="text-gray-500 animate-pulse">
                  {'>>'} Waiting for server logs...
                </div>
              )}
              <div className="h-4" /> {/* Spacing at bottom for better scrolling */}
            </div>
          </Card>

          {/* Contributors */}
          <Card className="p-6 bg-gray-800 border-gray-700">
            <Title className="text-white">Contributors</Title>
            <div className="mt-4 overflow-auto max-h-[300px]">
              <table className="min-w-full divide-y divide-gray-700">
                <thead>
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Rank
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Wallet Address
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Contributions
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Avg. Impact
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                      Total Impact
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-700">
                  {contributors.map((contributor, index) => (
                    <tr 
                      key={contributor.client_id}
                      className={`
                        ${index === 0 ? 'bg-blue-900/30' : 'hover:bg-gray-700/50'}
                        transition-colors
                      `}
                    >
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        {index === 0 ? (
                          <Badge color="blue">
                            #{index + 1}
                          </Badge>
                        ) : (
                          <span className="text-gray-400">#{index + 1}</span>
                        )}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        <div className="flex items-center">
                          {index === 0 && (
                            <span className="mr-2">👑</span>
                          )}
                          <span className={`font-medium ${index === 0 ? 'text-blue-400' : 'text-gray-300'}`}>
                            {contributor.client_id}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-300">
                        {contributor.contribution}
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        <Badge color={contributor.average_impact > 0 ? 'green' : 'red'}>
                          {contributor.average_impact.toFixed(2)}%
                        </Badge>
                      </td>
                      <td className="px-4 py-3 whitespace-nowrap text-sm">
                        <span className={`
                          ${contributor.total_impact > 0 ? 'text-green-400' : 'text-red-400'}
                        `}>
                          {contributor.total_impact.toFixed(2)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              
              {contributors.length === 0 && (
                <div className="flex items-center justify-center h-[200px]">
                  <p className="text-gray-400">No contributors yet</p>
                </div>
              )}
            </div>
          </Card>

          {/* Token Distribution */}
          <Card className="p-6 bg-gray-800 border-gray-700">
            <Title className="text-white">Token Distribution</Title>
            <Subtitle className="text-gray-400">Next distribution: {nextDistTime}</Subtitle>
            
            {accountId ? (
              <div className="mt-4 p-4 bg-gray-900 rounded-lg">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400">Your Balance:</span>
                  <span className="text-green-400 font-mono">
                    {(Number(userBalance) / 1e24).toFixed(2)} FLT
                  </span>
                </div>
              </div>
            ) : (
              <div className="mt-4 p-4 bg-gray-900/50 rounded-lg text-center">
                <p className="text-gray-400">Connect wallet to view balance</p>
              </div>
            )}
            
            <div className="mt-4 space-y-4">
              {contributors.slice(0, 5).map((contributor) => (
                <div key={contributor.client_id} className="flex justify-between items-center">
                  <span className="text-sm font-medium text-gray-300">
                    {contributor.client_id}
                  </span>
                  <Badge color="green">
                    {(contributor.average_impact * 100).toFixed(2)}% impact
                  </Badge>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
