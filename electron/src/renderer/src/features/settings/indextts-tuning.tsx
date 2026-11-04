import { AudioLinesIcon } from 'lucide-react';
import { useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useTranslation } from 'react-i18next';
import { Button } from '@/components/ui/button';
import { apiJson } from '@/lib/api/client';
import { SettingsSection, SettingsRow } from './settings-layout';
import { useSettingsAction } from './use-settings-action';

type Field = 's2m_precision' | 'num_beams' | 'low_vram_split';
interface IndexTTSTuningState {
  s2m_precision: string;
  num_beams: number | string;
  low_vram_split: string;
  env_pinned: Field[];
  vram_total_gb: number;
  upstream_splits: boolean;
  restarted?: boolean;
}
const PATH = '/api/settings/perf/indextts';
// Defaults measured in srs.md section 6.3: fp16 S2M halves the time with the
// same automatic quality scores; beam 1 misread words, so 3 stays.
const RECOMMENDED: Partial<Record<Field, string | number>> = {
  s2m_precision: 'fp16',
  num_beams: 3,
};

// Per-machine IndexTTS settings for company workers; each machine saves its own.
export function IndexTTSTuning() {
  const { t } = useTranslation();
  const client = useQueryClient();
  const action = useSettingsAction();
  const [restarted, setRestarted] = useState<boolean | null>(null);
  const query = useQuery({
    queryKey: ['indextts-tuning'],
    queryFn: ({ signal }) => apiJson<IndexTTSTuningState>(PATH, { signal }),
  });
  const state = query.data;
  const pinned = state?.env_pinned ?? [];
  const choose = (field: Field, value: string | number) =>
    void action.run(async () => {
      const saved = await apiJson<IndexTTSTuningState>(PATH, {
        method: 'PUT',
        body: JSON.stringify({ [field]: value }),
      });
      client.setQueryData(['indextts-tuning'], saved);
      setRestarted(!!saved.restarted);
    });
  const rows: { field: Field; title: string; description: string; choices: (string | number)[] }[] =
    [
      {
        field: 's2m_precision',
        title: t('settings.indextts_precision'),
        description: t('settings.indextts_precision_desc'),
        choices: ['fp16', 'fp32'],
      },
      {
        field: 'num_beams',
        title: t('settings.indextts_beams'),
        description: t('settings.indextts_beams_desc'),
        choices: [3, 1],
      },
      {
        field: 'low_vram_split',
        title: t('settings.indextts_split'),
        description: t('settings.indextts_split_desc'),
        choices: ['auto', 'off', 'on'],
      },
    ];
  const label = (field: Field, value: string | number) => {
    const text =
      value === 'auto'
        ? t('performanceProfile.auto')
        : value === 'off'
          ? t('cloneEmotion.off')
          : value === 'on'
            ? t('pronunciation.enabled')
            : String(value);
    return RECOMMENDED[field] === value ? `${text} (${t('firstrun.chip_recommended')})` : text;
  };
  return (
    <SettingsSection icon={AudioLinesIcon} title={t('settings.indextts_title')}>
      <p className="p-4 text-sm text-muted-foreground">{t('settings.indextts_desc')}</p>
      {rows.map(({ field, title, description, choices }) => (
        <SettingsRow key={field} id={'indextts-' + field} title={title} description={description}>
          {choices.map((value) => (
            <Button
              key={value}
              size="sm"
              variant={state?.[field] === value ? 'secondary' : 'ghost'}
              aria-pressed={state?.[field] === value}
              disabled={!state || action.busy || pinned.includes(field)}
              onClick={() => choose(field, value)}
            >
              {label(field, value)}
            </Button>
          ))}
        </SettingsRow>
      ))}
      {state?.upstream_splits && state.low_vram_split === 'auto' && (
        <p className="p-4 text-sm text-muted-foreground">
          {t('settings.indextts_split_note', { gb: state.vram_total_gb })}
        </p>
      )}
      {pinned.length > 0 && (
        <p className="p-4 text-sm text-muted-foreground">
          {t('settings.generate_timeout_shadowed_note')}
        </p>
      )}
      {restarted !== null && (
        <p role="status" className="p-4 text-sm text-muted-foreground">
          {t(restarted ? 'settings.indextts_saved_now' : 'settings.indextts_saved_later')}
        </p>
      )}
      {(action.error || query.isError) && (
        <p role="alert" className="p-4 text-sm text-destructive">
          {t('common.error')}{' '}
          <Button
            size="sm"
            variant="ghost"
            onClick={() => void query.refetch().then(() => action.clearError())}
          >
            {t('common.retry')}
          </Button>
        </p>
      )}
    </SettingsSection>
  );
}
