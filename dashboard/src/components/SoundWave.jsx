import { useState, useMemo, useRef, useEffect } from 'react'
import {
  Alert,
  AlertDescription,
  AlertIcon,
  Badge,
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Flex,
  Grid,
  GridItem,
  Heading,
  HStack,
  Select,
  SimpleGrid,
  Slider,
  SliderFilledTrack,
  SliderMark,
  SliderThumb,
  SliderTrack,
  Stat,
  StatHelpText,
  StatLabel,
  StatNumber,
  Switch,
  Text,
  useColorModeValue,
  VStack,
} from '@chakra-ui/react'
import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts'

// --- JS port of GenerateSoundWave from UE5_scriptsv2.cpp ---

const SAMPLE_RATE = 44100
const DISPLAY_SAMPLES = 600 // max points shown in chart

/**
 * Generate PCM samples for a pure sine wave tone.
 * Matches the C++ GenerateSoundWave logic (amplitude = 32767 * sin(2π*f*i/SR)).
 */
function generatePCM(frequency, duration, waveform = 'sine') {
  const numSamples = Math.round(SAMPLE_RATE * duration)
  const step = Math.max(1, Math.floor(numSamples / DISPLAY_SAMPLES))
  const points = []

  for (let i = 0; i < numSamples; i += step) {
    const t = i / SAMPLE_RATE
    let amplitude

    switch (waveform) {
      case 'square':
        amplitude = Math.sign(Math.sin(2 * Math.PI * frequency * t))
        break
      case 'sawtooth':
        amplitude = 2 * ((frequency * t) % 1) - 1
        break
      case 'triangle': {
        const phase = (frequency * t) % 1
        amplitude = phase < 0.5 ? 4 * phase - 1 : 3 - 4 * phase
        break
      }
      default: // sine
        amplitude = Math.sin(2 * Math.PI * frequency * t)
    }

    const pcm = Math.round(32767 * amplitude)
    points.push({
      t: parseFloat(t.toFixed(5)),
      pcm,
      normalized: parseFloat(amplitude.toFixed(4)),
      sampleIdx: i,
    })
  }
  return points
}

/**
 * Generate sound frequency based on quantum result modifier (from ExecuteIntegratedLogic in UE5).
 * freq = quantumResult * 1000 + 440
 */
function quantumFrequency(quantumResult) {
  return quantumResult * 1000 + 440
}

// Simple Web Audio API tone generator
function playTone(frequency, duration, waveform, volume) {
  if (typeof window === 'undefined') return
  const ctx = new (window.AudioContext || window.webkitAudioContext)()
  const osc = ctx.createOscillator()
  const gain = ctx.createGain()

  osc.type = waveform
  osc.frequency.setValueAtTime(frequency, ctx.currentTime)
  gain.gain.setValueAtTime(volume, ctx.currentTime)
  gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + duration)

  osc.connect(gain)
  gain.connect(ctx.destination)
  osc.start(ctx.currentTime)
  osc.stop(ctx.currentTime + duration)
}

function LabelledSlider({ label, value, min, max, step, onChange, colorScheme = 'purple', format }) {
  const fmt = format ?? (v => v.toFixed(2))
  return (
    <Box>
      <Flex justify="space-between" mb={1}>
        <Text fontSize="xs" color="gray.500">{label}</Text>
        <Text fontSize="xs" fontFamily="mono" fontWeight="semibold">{fmt(value)}</Text>
      </Flex>
      <Slider min={min} max={max} step={step} value={value} onChange={onChange} colorScheme={colorScheme}>
        <SliderTrack><SliderFilledTrack /></SliderTrack>
        <SliderThumb />
      </Slider>
    </Box>
  )
}

// Musical note frequencies
const NOTES = {
  'A4 (440 Hz)': 440,
  'C4 (261.6 Hz)': 261.63,
  'E4 (329.6 Hz)': 329.63,
  'G4 (392 Hz)': 392,
  'C5 (523.3 Hz)': 523.25,
  'A3 (220 Hz)': 220,
}

export default function SoundWave() {
  const [frequency, setFrequency] = useState(440)
  const [duration, setDuration] = useState(0.02)
  const [waveform, setWaveform] = useState('sine')
  const [volume, setVolume] = useState(0.3)
  const [showPCM, setShowPCM] = useState(false)
  const [quantumResult, setQuantumResult] = useState(0)

  const cardBg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.600')
  const gridColor = useColorModeValue('#e2e8f0', '#2d3748')
  const axisColor = useColorModeValue('#718096', '#a0aec0')

  const waveData = useMemo(
    () => generatePCM(frequency, duration, waveform),
    [frequency, duration, waveform]
  )

  const period = 1 / frequency
  const wavelengthMs = (period * 1000).toFixed(3)
  const numCycles = (frequency * duration).toFixed(1)
  const totalSamples = Math.round(SAMPLE_RATE * duration)
  const quantumFreq = quantumFrequency(quantumResult)

  return (
    <Grid templateColumns={{ base: '1fr', lg: '280px 1fr' }} gap={4}>
      {/* Controls */}
      <GridItem>
        <VStack spacing={3} align="stretch">
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">Wave Parameters</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <VStack spacing={4} align="stretch">
                <Box>
                  <Text fontSize="xs" color="gray.500" mb={1}>Preset Notes</Text>
                  <Select
                    size="sm"
                    onChange={e => setFrequency(parseFloat(e.target.value))}
                  >
                    {Object.entries(NOTES).map(([label, freq]) => (
                      <option key={freq} value={freq}>{label}</option>
                    ))}
                  </Select>
                </Box>

                <LabelledSlider
                  label="Frequency (Hz)"
                  value={frequency}
                  min={20}
                  max={2000}
                  step={1}
                  onChange={setFrequency}
                  colorScheme="purple"
                  format={v => `${v.toFixed(0)} Hz`}
                />

                <LabelledSlider
                  label="Duration (s)"
                  value={duration}
                  min={0.005}
                  max={0.1}
                  step={0.001}
                  onChange={setDuration}
                  colorScheme="blue"
                  format={v => `${v.toFixed(3)} s`}
                />

                <LabelledSlider
                  label="Volume"
                  value={volume}
                  min={0.01}
                  max={1}
                  step={0.01}
                  onChange={setVolume}
                  colorScheme="teal"
                />

                <Box>
                  <Text fontSize="xs" color="gray.500" mb={1}>Waveform Type</Text>
                  <Select
                    size="sm"
                    value={waveform}
                    onChange={e => setWaveform(e.target.value)}
                  >
                    <option value="sine">Sine</option>
                    <option value="square">Square</option>
                    <option value="sawtooth">Sawtooth</option>
                    <option value="triangle">Triangle</option>
                  </Select>
                </Box>

                <HStack justify="space-between">
                  <Text fontSize="xs">Show PCM int16 Values</Text>
                  <Switch colorScheme="purple" isChecked={showPCM} onChange={e => setShowPCM(e.target.checked)} size="sm" />
                </HStack>

                <Button
                  colorScheme="purple"
                  size="sm"
                  onClick={() => playTone(frequency, Math.min(duration * 50, 2), waveform, volume)}
                >
                  Play Tone
                </Button>
              </VStack>
            </CardBody>
          </Card>

          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">Quantum Integration</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <VStack spacing={3} align="stretch">
                <Text fontSize="xs" color="gray.500">
                  From <Text as="span" fontFamily="mono">ExecuteIntegratedLogic()</Text>:
                  freq = qResult × 1000 + 440
                </Text>
                <LabelledSlider
                  label="Quantum Result (0 or 1)"
                  value={quantumResult}
                  min={0}
                  max={1}
                  step={1}
                  onChange={setQuantumResult}
                  colorScheme="orange"
                  format={v => v.toFixed(0)}
                />
                <Flex justify="space-between" align="center">
                  <Text fontSize="xs" color="gray.500">Derived Frequency:</Text>
                  <Badge colorScheme="orange" fontFamily="mono">{quantumFreq} Hz</Badge>
                </Flex>
                <Button
                  colorScheme="orange"
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    setFrequency(quantumFreq)
                    playTone(quantumFreq, Math.min(duration * 50, 2), waveform, volume)
                  }}
                >
                  Apply & Play
                </Button>
              </VStack>
            </CardBody>
          </Card>
        </VStack>
      </GridItem>

      {/* Visualization */}
      <GridItem>
        <VStack spacing={4} align="stretch">
          {/* Stats */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Flex justify="space-between" align="center">
                <Heading size="sm">PCM Waveform — {frequency} Hz {waveform}</Heading>
                <Badge colorScheme="purple">{waveData.length} display pts</Badge>
              </Flex>
            </CardHeader>
            <CardBody pt={2}>
              <SimpleGrid columns={{ base: 2, md: 4 }} spacing={3} mb={4}>
                <Stat size="sm">
                  <StatLabel fontSize="xs">Frequency</StatLabel>
                  <StatNumber fontSize="md" fontFamily="mono">{frequency} Hz</StatNumber>
                </Stat>
                <Stat size="sm">
                  <StatLabel fontSize="xs">Period</StatLabel>
                  <StatNumber fontSize="md" fontFamily="mono">{wavelengthMs} ms</StatNumber>
                </Stat>
                <Stat size="sm">
                  <StatLabel fontSize="xs">Total Samples</StatLabel>
                  <StatNumber fontSize="md" fontFamily="mono">{totalSamples.toLocaleString()}</StatNumber>
                  <StatHelpText fontSize="xs">@ {SAMPLE_RATE / 1000}k SR</StatHelpText>
                </Stat>
                <Stat size="sm">
                  <StatLabel fontSize="xs">Cycles</StatLabel>
                  <StatNumber fontSize="md" fontFamily="mono">{numCycles}</StatNumber>
                  <StatHelpText fontSize="xs">in {duration.toFixed(3)} s</StatHelpText>
                </Stat>
              </SimpleGrid>

              <Box h="280px">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={waveData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <defs>
                      <linearGradient id="waveGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#805AD5" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#805AD5" stopOpacity={0.0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis
                      dataKey="t"
                      tick={{ fontSize: 10, fill: axisColor }}
                      tickFormatter={v => `${(v * 1000).toFixed(1)}ms`}
                    />
                    <YAxis
                      dataKey={showPCM ? 'pcm' : 'normalized'}
                      tick={{ fontSize: 10, fill: axisColor }}
                      domain={showPCM ? [-32768, 32767] : [-1.1, 1.1]}
                      tickFormatter={v => showPCM ? v.toLocaleString() : v.toFixed(1)}
                    />
                    <ChartTooltip
                      contentStyle={{ backgroundColor: cardBg, border: `1px solid ${borderColor}`, borderRadius: '8px', fontSize: '11px' }}
                      formatter={(v, name) => [
                        showPCM ? v.toLocaleString() : v.toFixed(4),
                        showPCM ? 'PCM int16' : 'Amplitude',
                      ]}
                      labelFormatter={t => `t = ${(t * 1000).toFixed(3)} ms`}
                    />
                    <ReferenceLine y={0} stroke={axisColor} strokeDasharray="2 2" />
                    <Area
                      type="monotone"
                      dataKey={showPCM ? 'pcm' : 'normalized'}
                      stroke="#805AD5"
                      fill="url(#waveGrad)"
                      dot={false}
                      strokeWidth={1.5}
                      name={showPCM ? 'PCM int16' : 'Amplitude'}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Box>
            </CardBody>
          </Card>

          <Alert status="info" borderRadius="lg">
            <AlertIcon />
            <Box>
              <AlertDescription fontSize="xs">
                Replicates <Text as="span" fontFamily="mono">GenerateSoundWave()</Text> from UE5_scriptsv2.cpp.
                PCM values = 32767 × sin(2π·f·i/44100), packed as int16 little-endian bytes.
                The "Play Tone" button uses the Web Audio API oscillator (same waveform logic).
              </AlertDescription>
            </Box>
          </Alert>
        </VStack>
      </GridItem>
    </Grid>
  )
}
