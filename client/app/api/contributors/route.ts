import { NextResponse } from 'next/server';
import axios from 'axios';
import { config } from '@/lib/config';

export async function GET() {
  try {
    const response = await axios.get(`${config.apiBaseUrl}/contributors`);
    return NextResponse.json(response.data);
  } catch (error) {
    console.error('Error fetching contributors:', error);
    return NextResponse.json(
      { error: 'Failed to fetch contributors' },
      { status: 500 }
    );
  }
} 