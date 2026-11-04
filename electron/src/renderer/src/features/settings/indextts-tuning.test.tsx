import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { afterEach, expect, it, vi } from 'vitest';

const mock = vi.hoisted(() => ({ api: vi.fn() }));
vi.mock('@/lib/api/client', () => ({ apiJson: mock.api }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }));
import { IndexTTSTuning } from './indextts-tuning';

const PATH = '/api/settings/perf/indextts';

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

function renderTuning() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <IndexTTSTuning />
    </QueryClientProvider>,
  );
}

it('saves one field and reports that the sidecar restarted', async () => {
  let state = {
    s2m_precision: 'fp16',
    num_beams: 3,
    low_vram_split: 'auto',
    env_pinned: [] as string[],
    vram_total_gb: 8,
    upstream_splits: true,
  };
  mock.api.mockImplementation(async (path: string, init?: RequestInit) => {
    expect(path).toBe(PATH);
    if (init?.method === 'PUT') {
      state = { ...state, ...JSON.parse(String(init.body)) };
      return { ...state, restarted: true };
    }
    return state;
  });
  renderTuning();
  expect(await screen.findByText('settings.indextts_split_note')).toBeInTheDocument();
  const off = screen.getByRole('button', { name: 'cloneEmotion.off' });
  await waitFor(() => expect(off).toBeEnabled());
  expect(screen.getByRole('button', { name: 'fp16 (firstrun.chip_recommended)' })).toHaveAttribute(
    'aria-pressed',
    'true',
  );
  fireEvent.click(off);
  expect(await screen.findByText('settings.indextts_saved_now')).toBeInTheDocument();
  expect(mock.api).toHaveBeenCalledWith(PATH, {
    method: 'PUT',
    body: JSON.stringify({ low_vram_split: 'off' }),
  });
  expect(off).toHaveAttribute('aria-pressed', 'true');
  expect(screen.queryByText('settings.indextts_split_note')).not.toBeInTheDocument();
});

it('locks a field pinned by an environment variable', async () => {
  mock.api.mockResolvedValue({
    s2m_precision: 'fp16',
    num_beams: 1,
    low_vram_split: 'auto',
    env_pinned: ['num_beams'],
    vram_total_gb: 24,
    upstream_splits: false,
  });
  renderTuning();
  expect(await screen.findByText('settings.generate_timeout_shadowed_note')).toBeInTheDocument();
  const beam1 = screen.getByRole('button', { name: '1' });
  expect(beam1).toBeDisabled();
  expect(beam1).toHaveAttribute('aria-pressed', 'true');
  expect(screen.getByRole('button', { name: 'fp32' })).toBeEnabled();
});
