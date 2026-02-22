import { useState, useCallback, useMemo } from 'react'
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
  Code,
  Divider,
  Flex,
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
  Table,
  TableContainer,
  Tbody,
  Td,
  Text,
  Th,
  Thead,
  Tooltip,
  Tr,
  useColorModeValue,
  VStack,
} from '@chakra-ui/react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
  XAxis,
  YAxis,
} from 'recharts'

// --- JS ports of HFP/DFA from UE5_scripts.cpp ---

// Seeded pseudo-random (same as TimeSeries.jsx)
function makePRNG(seed) {
  let s = seed >>> 0
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0
    return s / 0xffffffff
  }
}

/**
 * CalculateHyperrootFluxParameter
 * HFP = Σ_i Σ_j R * exp(H[i][j] * Π_k S[i][j][k])
 */
function calculateHFP(R, H, S) {
  let hfp = 0
  for (let i = 0; i < H.length; i++) {
    for (let j = 0; j < H[i].length; j++) {
      let sProd = 1
      for (let k = 0; k < S[i][j].length; k++) {
        sProd *= S[i][j][k]
      }
      hfp += R * Math.exp(H[i][j] * sProd)
    }
  }
  return hfp
}

/**
 * CalculateHyperrootFluxParameterExpanded (vectorized variant)
 * HFP = Σ_i R * exp(Σ_j H[i][j] * Π_k S[i][j][k])
 */
function calculateHFPExpanded(R, H, S) {
  let hfp = 0
  for (let i = 0; i < H.length; i++) {
    let sumExp = 0
    for (let j = 0; j < H[i].length; j++) {
      let sProd = 1
      for (let k = 0; k < S[i][j].length; k++) {
        sProd *= S[i][j][k]
      }
      sumExp += H[i][j] * sProd
    }
    hfp += R * Math.exp(sumExp)
  }
  return hfp
}

/**
 * CalculateDigitalFluxAmbiance
 * DFA = λ * (HFP * I^a + D^b * E^c)
 */
function calculateDFA(HFP, lambda, I, D, E, a, b, c) {
  return lambda * (HFP * Math.pow(I, a) + Math.pow(D, b) * Math.pow(E, c))
}

function randomMatrix2D(rows, cols, rand) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => parseFloat(rand().toFixed(4)))
  )
}

function randomMatrix3D(n, m, p, rand) {
  return Array.from({ length: n }, () =>
    Array.from({ length: m }, () =>
      Array.from({ length: p }, () => parseFloat(rand().toFixed(4)))
    )
  )
}

function LabelledSlider({ label, value, min, max, step, onChange, colorScheme = 'purple', format }) {
  const fmt = format ?? (v => v.toFixed(3))
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

const SWEEP_STEPS = 20

export default function HFPCalculator() {
  // HFP parameters
  const [R, setR] = useState(1.5)
  const [n, setN] = useState(2)
  const [m, setM] = useState(3)
  const [p, setP] = useState(2)
  const [seed, setSeed] = useState(42)
  const [useExpanded, setUseExpanded] = useState(false)

  // DFA parameters
  const [lambda, setLambda] = useState(1.5)
  const [I, setI] = useState(2.0)
  const [D, setD] = useState(1.2)
  const [E, setE] = useState(3.0)
  const [a, setA] = useState(1.2)
  const [b, setB] = useState(0.8)
  const [c, setC] = useState(1.5)

  const cardBg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.600')
  const gridColor = useColorModeValue('#e2e8f0', '#2d3748')
  const axisColor = useColorModeValue('#718096', '#a0aec0')

  const { H, S, hfp, hfpExpanded } = useMemo(() => {
    const rand = makePRNG(seed)
    const H = randomMatrix2D(n, m, rand)
    const S = randomMatrix3D(n, m, p, rand)
    return {
      H,
      S,
      hfp: calculateHFP(R, H, S),
      hfpExpanded: calculateHFPExpanded(R, H, S),
    }
  }, [R, n, m, p, seed])

  const dfa = useMemo(
    () => calculateDFA(hfp, lambda, I, D, E, a, b, c),
    [hfp, lambda, I, D, E, a, b, c]
  )

  // Sweep R from 0.1 to 5.0 to chart HFP vs R
  const sweepData = useMemo(() => {
    const rand = makePRNG(seed)
    const H = randomMatrix2D(n, m, rand)
    const S = randomMatrix3D(n, m, p, rand)
    return Array.from({ length: SWEEP_STEPS }, (_, i) => {
      const rVal = 0.25 + (i / (SWEEP_STEPS - 1)) * 4.75
      return {
        R: parseFloat(rVal.toFixed(2)),
        HFP: parseFloat(calculateHFP(rVal, H, S).toFixed(4)),
        HFP_exp: parseFloat(calculateHFPExpanded(rVal, H, S).toFixed(4)),
      }
    })
  }, [n, m, p, seed])

  // Sweep lambda for DFA
  const dfaSweep = useMemo(() => {
    return Array.from({ length: SWEEP_STEPS }, (_, i) => {
      const lVal = 0.1 + (i / (SWEEP_STEPS - 1)) * 4.9
      return {
        lambda: parseFloat(lVal.toFixed(2)),
        DFA: parseFloat(calculateDFA(hfp, lVal, I, D, E, a, b, c).toFixed(4)),
      }
    })
  }, [hfp, I, D, E, a, b, c])

  const hInline = H.map(row => `[${row.join(', ')}]`).join('\n')

  return (
    <Grid templateColumns={{ base: '1fr', lg: '300px 1fr' }} gap={4}>
      {/* Controls */}
      <GridItem>
        <VStack spacing={3} align="stretch">
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">HFP Parameters</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <VStack spacing={3} align="stretch">
                <LabelledSlider label="R — Root Hyperflux" value={R} min={0.1} max={5} step={0.05} onChange={setR} />
                <Box>
                  <Text fontSize="xs" color="gray.500" mb={1}>n — Multitory Levels</Text>
                  <NumberInput min={1} max={6} value={n} onChange={v => setN(parseInt(v))} size="sm">
                    <NumberInputField />
                    <NumberInputStepper><NumberIncrementStepper /><NumberDecrementStepper /></NumberInputStepper>
                  </NumberInput>
                </Box>
                <Box>
                  <Text fontSize="xs" color="gray.500" mb={1}>m — Dense Hyperparameters</Text>
                  <NumberInput min={1} max={6} value={m} onChange={v => setM(parseInt(v))} size="sm">
                    <NumberInputField />
                    <NumberInputStepper><NumberIncrementStepper /><NumberDecrementStepper /></NumberInputStepper>
                  </NumberInput>
                </Box>
                <Box>
                  <Text fontSize="xs" color="gray.500" mb={1}>p — Subbase Sector Dims</Text>
                  <NumberInput min={1} max={6} value={p} onChange={v => setP(parseInt(v))} size="sm">
                    <NumberInputField />
                    <NumberInputStepper><NumberIncrementStepper /><NumberDecrementStepper /></NumberInputStepper>
                  </NumberInput>
                </Box>
                <Box>
                  <Text fontSize="xs" color="gray.500" mb={1}>Random Seed</Text>
                  <NumberInput min={1} max={9999} value={seed} onChange={v => setSeed(parseInt(v))} size="sm">
                    <NumberInputField />
                    <NumberInputStepper><NumberIncrementStepper /><NumberDecrementStepper /></NumberInputStepper>
                  </NumberInput>
                </Box>
              </VStack>
            </CardBody>
          </Card>

          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">DFA Parameters</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <VStack spacing={3} align="stretch">
                <LabelledSlider label="λ (lambda)" value={lambda} min={0.1} max={5} step={0.05} onChange={setLambda} colorScheme="blue" />
                <LabelledSlider label="I — Intensity" value={I} min={0.1} max={10} step={0.1} onChange={setI} colorScheme="blue" />
                <LabelledSlider label="D — Density" value={D} min={0.1} max={5} step={0.05} onChange={setD} colorScheme="teal" />
                <LabelledSlider label="E — External Influences" value={E} min={0.1} max={10} step={0.1} onChange={setE} colorScheme="teal" />
                <Divider />
                <LabelledSlider label="a — exponent for I" value={a} min={0.1} max={3} step={0.05} onChange={setA} colorScheme="orange" />
                <LabelledSlider label="b — exponent for D" value={b} min={0.1} max={3} step={0.05} onChange={setB} colorScheme="orange" />
                <LabelledSlider label="c — exponent for E" value={c} min={0.1} max={3} step={0.05} onChange={setC} colorScheme="orange" />
              </VStack>
            </CardBody>
          </Card>
        </VStack>
      </GridItem>

      {/* Results */}
      <GridItem>
        <VStack spacing={4} align="stretch">
          {/* Result stats */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">Computed Values</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <SimpleGrid columns={{ base: 2, md: 4 }} spacing={4}>
                <Stat>
                  <StatLabel fontSize="xs">HFP (standard)</StatLabel>
                  <StatNumber fontSize="lg" fontFamily="mono" color="purple.400">
                    {hfp.toFixed(4)}
                  </StatNumber>
                  <StatHelpText fontSize="xs">R={R} n={n} m={m} p={p}</StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel fontSize="xs">HFP (expanded)</StatLabel>
                  <StatNumber fontSize="lg" fontFamily="mono" color="blue.400">
                    {hfpExpanded.toFixed(4)}
                  </StatNumber>
                  <StatHelpText fontSize="xs">vectorized variant</StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel fontSize="xs">DFA</StatLabel>
                  <StatNumber fontSize="lg" fontFamily="mono" color="teal.400">
                    {dfa.toFixed(4)}
                  </StatNumber>
                  <StatHelpText fontSize="xs">λ={lambda} I={I} D={D} E={E}</StatHelpText>
                </Stat>
                <Stat>
                  <StatLabel fontSize="xs">HFP Δ (standard vs expanded)</StatLabel>
                  <StatNumber fontSize="lg" fontFamily="mono" color="orange.400">
                    {(hfp - hfpExpanded).toFixed(6)}
                  </StatNumber>
                  <StatHelpText fontSize="xs">difference between variants</StatHelpText>
                </Stat>
              </SimpleGrid>

              {/* Formula display */}
              <Alert status="info" mt={4} borderRadius="md">
                <AlertIcon />
                <Box>
                  <AlertDescription fontSize="xs" fontFamily="mono">
                    HFP = Σᵢ Σⱼ R·exp(H[i][j] · Πₖ S[i][j][k])
                    <br />
                    DFA = λ · (HFP · I<sup>a</sup> + D<sup>b</sup> · E<sup>c</sup>)
                  </AlertDescription>
                </Box>
              </Alert>
            </CardBody>
          </Card>

          {/* HFP vs R sweep chart */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Flex justify="space-between" align="center">
                <Heading size="sm">HFP vs R (sweep)</Heading>
                <Badge colorScheme="purple">n={n}, m={m}, p={p}</Badge>
              </Flex>
            </CardHeader>
            <CardBody pt={2}>
              <Box h="220px">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={sweepData} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis dataKey="R" tick={{ fontSize: 11, fill: axisColor }} />
                    <YAxis tick={{ fontSize: 11, fill: axisColor }} />
                    <ChartTooltip
                      contentStyle={{ backgroundColor: cardBg, border: `1px solid ${borderColor}`, borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Line type="monotone" dataKey="HFP" stroke="#805AD5" dot={false} strokeWidth={2} name="HFP" />
                    <Line type="monotone" dataKey="HFP_exp" stroke="#3182CE" dot={false} strokeWidth={2} strokeDasharray="4 2" name="HFP expanded" />
                  </LineChart>
                </ResponsiveContainer>
              </Box>
            </CardBody>
          </Card>

          {/* DFA vs lambda sweep chart */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">DFA vs λ (sweep)</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <Box h="200px">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={dfaSweep} margin={{ top: 5, right: 10, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke={gridColor} />
                    <XAxis dataKey="lambda" tick={{ fontSize: 11, fill: axisColor }} />
                    <YAxis tick={{ fontSize: 11, fill: axisColor }} />
                    <ChartTooltip
                      contentStyle={{ backgroundColor: cardBg, border: `1px solid ${borderColor}`, borderRadius: '8px', fontSize: '12px' }}
                    />
                    <Bar dataKey="DFA" name="DFA" radius={[3, 3, 0, 0]}>
                      {dfaSweep.map((_, i) => (
                        <Cell key={i} fill={`hsl(${260 - i * 8}, 70%, 60%)`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Box>
            </CardBody>
          </Card>

          {/* H matrix display */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">H Matrix (Base Dense Hyperparameters)</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <TableContainer>
                <Table size="sm" variant="simple">
                  <Thead>
                    <Tr>
                      <Th fontSize="xs">i \ j</Th>
                      {Array.from({ length: m }, (_, j) => (
                        <Th key={j} fontSize="xs" isNumeric>j={j}</Th>
                      ))}
                    </Tr>
                  </Thead>
                  <Tbody>
                    {H.map((row, i) => (
                      <Tr key={i}>
                        <Td fontSize="xs" fontWeight="semibold" color="purple.400">i={i}</Td>
                        {row.map((val, j) => (
                          <Td key={j} fontSize="xs" fontFamily="mono" isNumeric>{val.toFixed(4)}</Td>
                        ))}
                      </Tr>
                    ))}
                  </Tbody>
                </Table>
              </TableContainer>
            </CardBody>
          </Card>
        </VStack>
      </GridItem>
    </Grid>
  )
}
