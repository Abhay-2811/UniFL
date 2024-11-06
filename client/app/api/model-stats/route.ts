import { NextResponse } from 'next/server';
import axios from 'axios';
import { config } from '@/lib/config';

export async function GET() {
  try {
    const response = await axios.get(`${config.apiBaseUrl}/stats`, {
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      }
    });
    return NextResponse.json(response.data);
  } catch (error) {
    console.error('Error fetching stats:', error);
    return NextResponse.json(
      { error: 'Failed to fetch model stats' },
      { status: 500 }
    );
  }
} 