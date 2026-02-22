import { useState, useCallback } from 'react'
import {
  Badge,
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Flex,
  FormControl,
  FormLabel,
  Grid,
  GridItem,
  Heading,
  HStack,
  NumberDecrementStepper,
  NumberIncrementStepper,
  NumberInput,
  NumberInputField,
  NumberInputStepper,
  SimpleGrid,
  Slider,
  SliderFilledTrack,
  SliderThumb,
  SliderTrack,
  Stat,
  StatHelpText,
  StatLabel,
  StatNumber,
  Switch,
  Text,
  Tooltip,
  useColorModeValue,
  VStack,
} from '@chakra-ui/react'
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts'

// --- JS ports of UE5 time series algorithms ---

function pseudoRandom(seed) {
  let s = seed
  return () => {
    s = (s * 1664525 + 1013904223) & 0xffffffff
    return ((s >>> 0) / 0xffffffff) * 2 - 1 // [-1, 1]
  }
}

function generateSineData(timesteps, noiseAmp, seed = 42) {
  const rand = pseudoRandom(seed)
  return Array.from({ length: timesteps }, (_, i) => {
    const x = (i / (timesteps - 1)) * 2 * Math.PI
    const signal = Math.sin(x)
    const noise = rand() * noiseAmp
    return {
      t: i,
      signal: parseFloat(signal.toFixed(4)),
      noisy: parseFloat((signal + noise).toFixed(4)),
    }
  })
}

// Simple LSTM-like forecast: weighted average with exponential decay
function lstmForecast(data, lookback = 10) {
  const values = data.map(d => d.noisy)
  const forecast = values.map((_, i) => {
    if (i < lookback) return null
    let weightSum = 0
    let valSum = 0
    for (let j = 1; j <= lookback; j++) {
      const w = Math.exp(-j * 0.3)
      weightSum += w
      valSum += values[i - j] * w
    }
    return parseFloat((valSum / weightSum).toFixed(4))
  })
  return data.map((d, i) => ({ ...d, forecast: forecast[i] }))
}

function generateHexagonalData(timesteps, dataDim) {
  const rand = pseudoRandom(99)
  return Array.from({ length: timesteps }, (_, t) => {
    const point = { t }
    for (let d = 0; d < dataDim; d++) {
      point[`dim${d}`] = parseFloat(((rand() + 1) / 2).toFixed(4))
    }
    return point
  })
}

const COLORS = ['#805AD5', '#3182CE', '#38A169', '#DD6B20', '#E53E3E', '#D69E2E']

export default function TimeSeries() {
  const [timesteps, setTimesteps] = useState(100)
  const [noiseAmp, setNoiseAmp] = useState(0.1)
  const [lookback, setLookback] = useState(10)
  const [showForecast, setShowForecast] = useState(true)
  const [showSignal, setShowSignal] = useState(true)
  const [hexDims, setHexDims] = useState(3)
  const [activeTab, setActiveTab] = useState('sine') // 'sine' | 'hex'

  const cardBg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.600')
  const gridColor = useColorModeValue('#e2e8f0', '#2d3748')
  const axisColor = useColorModeValue('#718096', '#a0aec0')

  const sineData = useCallback(
    () => lstmForecast(generateSineData(timesteps, noiseAmp), lookback),
    [timesteps, noiseAmp, lookback]
  )
  const hexData = useCallback(
    () => generateHexagonalData(timesteps, hexDims),
    [timesteps, hexDims]
  )

  const [sineChartData, setSineChartData] = useState(sineData)
  const [hexChartData, setHexChartData] = useState(hexData)

  const regenerate = () => {
    setSineChartData(sineData())
    setHexChartData(hexData())
  }

  const forecastValues = sineChartData.filter(d => d.forecast !== null).map(d => d.forecast)
  const forecastMSE =
    forecastValues.length > 0
      ? (
          forecastValues.reduce((acc, v, i) => {
            const actual = sineChartData[i + lookback]?.noisy ?? 0
            return acc + (v - actual) ** 2
          }, 0) / forecastValues.length
        ).toFixed(5)
      : 'N/A'

  return (
    <Grid templateColumns={{ base: '1fr', lg: '280px 1fr' }} gap={4}>
      {/* Controls */}
      <GridItem>
        <Card bg={cardBg} border="1px" borderColor={borderColor}>
          <CardHeader pb={2}>
            <Heading size="sm">Parameters</Heading>
          </CardHeader>
          <CardBody pt={2}>
            <VStack spacing={4} align="stretch">
              <FormControl>
                <FormLabel fontSize="xs" mb={1}>Timesteps</FormLabel>
                <NumberInput
                  min={20}
                  max={500}
                  value={timesteps}
                  onChange={val => setTimesteps(parseInt(val, 10))}
                  size="sm"
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>

              <FormControl>
                <Flex justify="space-between" mb={1}>
                  <FormLabel fontSize="xs" mb={0}>Noise Amplitude</FormLabel>
                  <Text fontSize="xs" fontFamily="mono">{noiseAmp.toFixed(2)}</Text>
                </Flex>
                <Slider
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={noiseAmp}
                  onChange={setNoiseAmp}
                  colorScheme="purple"
                >
                  <SliderTrack><SliderFilledTrack /></SliderTrack>
                  <SliderThumb />
                </Slider>
              </FormControl>

              <FormControl>
                <Flex justify="space-between" mb={1}>
                  <FormLabel fontSize="xs" mb={0}>LSTM Lookback</FormLabel>
                  <Text fontSize="xs" fontFamily="mono">{lookback}</Text>
                </Flex>
                <Slider
                  min={2}
                  max={40}
                  step={1}
                  value={lookback}
                  onChange={setLookback}
                  colorScheme="blue"
                >
                  <SliderTrack><SliderFilledTrack /></SliderTrack>
                  <SliderThumb />
                </Slider>
              </FormControl>

              <Divider />

              <FormControl>
                <FormLabel fontSize="xs" mb={1}>Hex Data Dimensions</FormLabel>
                <NumberInput
                  min={1}
                  max={6}
                  value={hexDims}
                  onChange={val => setHexDims(parseInt(val, 10))}
                  size="sm"
                >
                  <NumberInputField />
                  <NumberInputStepper>
                    <NumberIncrementStepper />
                    <NumberDecrementStepper />
                  </NumberInputStepper>
                </NumberInput>
              </FormControl>

              <Divider />

              <HStack justify="space-between">
                <Text fontSize="xs">Show True Signal</Text>
                <Switch
                  colorScheme="purple"
                  isChecked={showSignal}
                  onChange={e => setShowSignal(e.target.checked)}
                  size="sm"
                />
              </HStack>
              <HStack justify="space-between">
                <Text fontSize="xs">Show LSTM Forecast</Text>
                <Switch
                  colorScheme="blue"
                  isChecked={showForecast}
                  onChange={e => setShowForecast(e.target.checked)}
                  size="sm"
                />
              </HStack>

              <Divider />

              <Button colorScheme="purple" size="sm" onClick={regenerate}>
                Regenerate Data
              </Button>
            </VStack>
          </CardBody>
        </Card>
      </GridItem>

      {/* Charts */}
      <GridItem>
        <VStack spacing={4} align="stretch">
          {/* Tab switcher */}
          <HStack spacing={2}>
            <Button
              size="sm"
              variant={activeTab === 'sine' ? 'solid' : 'outline'}
              colorScheme="purple"
              onClick={() => setActiveTab('sine')}
            >
              Sine Wave + LSTM
            </Button>
            <Button
              size="sm"
              variant={activeTab === 'hex' ? 'solid' : 'outline'}
              colorScheme="blue"
              onClick={() => setActiveTab('hex')}
            >
              Hexagonal Data
            </Button>
          </HStack>

          {/* Sine + forecast chart */}
          {activeTab === 'sine' && (
            <Card bg={cardBg} border="1px" borderColor={borderColor}>
              <CardHeader pb={2}>
                <Flex justify="space-between" align="center">
                  <Heading size="sm">Synthetic Time Series + LSTM Forecast</Heading>
                  <Badge colorScheme="purple">{timesteps} steps</Badge>
                </Flex>
              </CardHeader>
              <CardBody pt={2}>
                <SimpleGrid columns={{ base: 2, md: 4 }} spacing={3} mb={4}>
                  <Stat size="sm">
                    <StatLabel fontSize="xs">Mean (noisy)</StatLabel>
                    <StatNumber fontSize="md" fontFamily="mono">
                      {(sineChartData.reduce((a, d) => a + d.noisy, 0) / sineChartData.length).toFixed(3)}
                    </StatNumber>
                  </Stat>
                  <Stat size="sm">
                    <StatLabel fontSize="xs">Noise Amp</StatLabel>
                    <StatNumber fontSize="md" fontFamily="mono">{noiseAmp.toFixed(2)}</StatNumber>
                  </Stat>
                  <Stat size="sm">
                    <StatLabel fontSize="xs">Forecast MSE</StatLabel>
                    <StatNumber fontSize="md" fontFamily="mono">{forecastMSE}</StatNumber>
                    <StatHelpText fontSize="xs">lookback={lookback}</StatHelpText>
                  </Stat>
                  <Stat size="sm">
                    <StatLabel fontSize="xs">Forecast Points</StatLabel>
                    <StatNumber fontSize="md" fontFamily="mono">{forecastValues.length}</StatNumber>
                  </Stat>
                </SimpleGrid>

                <Box h="300px">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={sineChartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                      <XAxis dataKey="t" tick={{ fontSize: 11, fill: axisColor }} />
                      <YAxis tick={{ fontSize: 11, fill: axisColor }} domain={[-1.8, 1.8]} />
                      <ChartTooltip
                        contentStyle={{
                          backgroundColor: cardBg,
                          border: `1px solid ${borderColor}`,
                          borderRadius: '8px',
                          fontSize: '12px',
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: '12px' }} />
                      <ReferenceLine y={0} stroke={axisColor} strokeDasharray="2 2" />
                      <Line
                        type="monotone"
                        dataKey="noisy"
                        stroke="#805AD5"
                        dot={false}
                        strokeWidth={1.5}
                        name="Noisy Signal"
                      />
                      {showSignal && (
                        <Line
                          type="monotone"
                          dataKey="signal"
                          stroke="#E2E8F0"
                          dot={false}
                          strokeWidth={1}
                          strokeDasharray="4 2"
                          name="True Signal"
                        />
                      )}
                      {showForecast && (
                        <Line
                          type="monotone"
                          dataKey="forecast"
                          stroke="#3182CE"
                          dot={false}
                          strokeWidth={2}
                          strokeDasharray="6 3"
                          name="LSTM Forecast"
                          connectNulls={false}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>
          )}

          {/* Hexagonal data chart */}
          {activeTab === 'hex' && (
            <Card bg={cardBg} border="1px" borderColor={borderColor}>
              <CardHeader pb={2}>
                <Flex justify="space-between" align="center">
                  <Heading size="sm">Hexagonal Encrypted Glyph Data</Heading>
                  <Badge colorScheme="blue">{hexDims} dim(s)</Badge>
                </Flex>
              </CardHeader>
              <CardBody pt={2}>
                <Box h="340px">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={hexChartData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                      <XAxis dataKey="t" tick={{ fontSize: 11, fill: axisColor }} />
                      <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: axisColor }} />
                      <ChartTooltip
                        contentStyle={{
                          backgroundColor: cardBg,
                          border: `1px solid ${borderColor}`,
                          borderRadius: '8px',
                          fontSize: '12px',
                        }}
                      />
                      <Legend wrapperStyle={{ fontSize: '12px' }} />
                      {Array.from({ length: hexDims }, (_, d) => (
                        <Line
                          key={d}
                          type="monotone"
                          dataKey={`dim${d}`}
                          stroke={COLORS[d % COLORS.length]}
                          dot={false}
                          strokeWidth={1.5}
                          name={`Glyph dim-${d}`}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </Box>
              </CardBody>
            </Card>
          )}
        </VStack>
      </GridItem>
    </Grid>
  )
}
