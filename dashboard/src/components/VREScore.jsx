import { useState, useMemo } from 'react'
import {
  Badge,
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
  CircularProgress,
  CircularProgressLabel,
  Divider,
  Flex,
  Grid,
  GridItem,
  Heading,
  HStack,
  Progress,
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
  Text,
  Tooltip,
  useColorModeValue,
  VStack,
} from '@chakra-ui/react'
import {
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip as ChartTooltip,
} from 'recharts'

// --- JS ports of VRE score functions from UE5_scripts.cpp ---

const quantumEnhancement = (power, reduction) => power * reduction
const blockchainSecurity = (strength, integrity) => strength + integrity
const hapticFeedback = (accuracy, comfort) => accuracy * comfort
const aiDynamicContent = (lr, feedback) => lr * feedback.reduce((a, b) => a + b, 0)
const narrativeAdaptation = (impacts, flexibility) =>
  impacts.reduce((a, b) => a + b, 0) * flexibility

const DEFAULT_COMPONENTS = {
  C: { label: 'Connectivity', color: 'purple', desc: 'QuantumEnhancement(QPP × ECR)' },
  S: { label: 'Security', color: 'blue', desc: 'BlockchainSecurity(strength + integrity)' },
  U: { label: 'User Experience', color: 'teal', desc: 'HapticFeedback(accuracy × comfort)' },
  AI: { label: 'AI Content', color: 'orange', desc: 'AIDynamicContent(LR × Σfeedback)' },
  E: { label: 'Env. Dynamics', color: 'green', desc: 'NarrativeAdaptation(Σimpact × flex)' },
}

const DEFAULT_WEIGHTS = { C: 0.1, S: 0.15, U: 0.2, AI: 0.25, E: 0.3 }

// Normalize a raw score to [0, 1] for display
function normalize(val, min = 0, max = 300) {
  return Math.min(1, Math.max(0, (val - min) / (max - min)))
}

function LabelledSlider({ label, value, min, max, step, onChange, colorScheme = 'purple', format }) {
  const fmt = format ?? (v => v.toFixed(2))
  return (
    <FormControlRow label={label} value={fmt(value)}>
      <Slider min={min} max={max} step={step} value={value} onChange={onChange} colorScheme={colorScheme}>
        <SliderTrack><SliderFilledTrack /></SliderTrack>
        <SliderThumb />
      </Slider>
    </FormControlRow>
  )
}

function FormControlRow({ label, value, children }) {
  return (
    <Box>
      <Flex justify="space-between" mb={1}>
        <Text fontSize="xs" color="gray.500">{label}</Text>
        <Text fontSize="xs" fontFamily="mono" fontWeight="semibold">{value}</Text>
      </Flex>
      {children}
    </Box>
  )
}

export default function VREScore() {
  // Connectivity (Quantum Enhancement)
  const [qpp, setQpp] = useState(100)     // QuantumProcessingPower
  const [ecr, setEcr] = useState(0.8)     // EnvironmentalComplexityReduction

  // Security (Blockchain)
  const [encStrength, setEncStrength] = useState(256)
  const [transIntegrity, setTransIntegrity] = useState(0.99)

  // User Experience (Haptic)
  const [sensAccuracy, setSensAccuracy] = useState(0.9)
  const [userComfort, setUserComfort] = useState(0.95)

  // AI Content
  const [aiLR, setAiLR] = useState(0.05)
  const [aiFeedback, setAiFeedback] = useState([0.8, 0.9, 1.0])

  // Environmental Dynamics
  const [envImpacts, setEnvImpacts] = useState([0.8, 0.9, 1.0])
  const [flexibility, setFlexibility] = useState(0.7)

  // Weights
  const [weights, setWeights] = useState(DEFAULT_WEIGHTS)

  const cardBg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.600')

  const components = useMemo(() => ({
    C: quantumEnhancement(qpp, ecr),
    S: blockchainSecurity(encStrength, transIntegrity),
    U: hapticFeedback(sensAccuracy, userComfort),
    AI: aiDynamicContent(aiLR, aiFeedback),
    E: narrativeAdaptation(envImpacts, flexibility),
  }), [qpp, ecr, encStrength, transIntegrity, sensAccuracy, userComfort, aiLR, aiFeedback, envImpacts, flexibility])

  const totalScore = useMemo(() => {
    return Object.keys(components).reduce((acc, key) => {
      return acc + components[key] * weights[key]
    }, 0)
  }, [components, weights])

  // Normalized for radar (0-100)
  const radarData = Object.entries(DEFAULT_COMPONENTS).map(([key, meta]) => ({
    subject: meta.label,
    value: Math.min(100, normalize(components[key], 0, key === 'S' ? 300 : 10) * 100),
    fullMark: 100,
  }))

  const totalNorm = Math.min(100, (totalScore / 100) * 100)

  const scoreColor =
    totalNorm >= 70 ? 'green' : totalNorm >= 40 ? 'yellow' : 'red'

  const resetDefaults = () => {
    setQpp(100); setEcr(0.8); setEncStrength(256); setTransIntegrity(0.99)
    setSensAccuracy(0.9); setUserComfort(0.95); setAiLR(0.05)
    setAiFeedback([0.8, 0.9, 1.0]); setEnvImpacts([0.8, 0.9, 1.0])
    setFlexibility(0.7); setWeights(DEFAULT_WEIGHTS)
  }

  return (
    <Grid templateColumns={{ base: '1fr', lg: '320px 1fr' }} gap={4}>
      {/* Controls */}
      <GridItem>
        <VStack spacing={3} align="stretch">
          {/* Component Inputs */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">Component Inputs</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <VStack spacing={3} align="stretch">
                <Text fontSize="xs" fontWeight="semibold" color="purple.400" textTransform="uppercase">
                  Connectivity — QuantumEnhancement
                </Text>
                <LabelledSlider label="Quantum Processing Power" value={qpp} min={1} max={200} step={1} onChange={setQpp} format={v => v.toFixed(0)} />
                <LabelledSlider label="Env. Complexity Reduction" value={ecr} min={0} max={1} step={0.01} onChange={setEcr} />

                <Divider />
                <Text fontSize="xs" fontWeight="semibold" color="blue.400" textTransform="uppercase">
                  Security — BlockchainSecurity
                </Text>
                <LabelledSlider label="Encryption Strength" value={encStrength} min={64} max={512} step={8} onChange={setEncStrength} colorScheme="blue" format={v => v.toFixed(0)} />
                <LabelledSlider label="Transaction Integrity" value={transIntegrity} min={0} max={1} step={0.01} onChange={setTransIntegrity} colorScheme="blue" />

                <Divider />
                <Text fontSize="xs" fontWeight="semibold" color="teal.400" textTransform="uppercase">
                  User Experience — HapticFeedback
                </Text>
                <LabelledSlider label="Sensory Input Accuracy" value={sensAccuracy} min={0} max={1} step={0.01} onChange={setSensAccuracy} colorScheme="teal" />
                <LabelledSlider label="User Comfort Level" value={userComfort} min={0} max={1} step={0.01} onChange={setUserComfort} colorScheme="teal" />

                <Divider />
                <Text fontSize="xs" fontWeight="semibold" color="orange.400" textTransform="uppercase">
                  AI Content — AIDynamicContent
                </Text>
                <LabelledSlider label="Learning Rate" value={aiLR} min={0.001} max={0.5} step={0.001} onChange={setAiLR} colorScheme="orange" />

                <Divider />
                <Text fontSize="xs" fontWeight="semibold" color="green.400" textTransform="uppercase">
                  Env. Dynamics — NarrativeAdaptation
                </Text>
                <LabelledSlider label="Story Flexibility Index" value={flexibility} min={0} max={1} step={0.01} onChange={setFlexibility} colorScheme="green" />
              </VStack>
            </CardBody>
          </Card>

          {/* Weights */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">Component Weights</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <VStack spacing={3} align="stretch">
                {Object.entries(weights).map(([key, w]) => (
                  <LabelledSlider
                    key={key}
                    label={`${DEFAULT_COMPONENTS[key].label} (${key})`}
                    value={w}
                    min={0.01}
                    max={0.5}
                    step={0.01}
                    onChange={v => setWeights(prev => ({ ...prev, [key]: v }))}
                    colorScheme={DEFAULT_COMPONENTS[key].color}
                  />
                ))}
              </VStack>
            </CardBody>
          </Card>

          <Button variant="outline" size="sm" onClick={resetDefaults}>
            Reset to Defaults
          </Button>
        </VStack>
      </GridItem>

      {/* Results */}
      <GridItem>
        <VStack spacing={4} align="stretch">
          {/* Score gauge */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">VRE Total Score</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <Flex align="center" justify="space-around" flexWrap="wrap" gap={4}>
                <CircularProgress
                  value={Math.min(100, totalNorm)}
                  size="140px"
                  thickness="10px"
                  color={`${scoreColor}.400`}
                  trackColor={useColorModeValue('gray.100', 'gray.700')}
                >
                  <CircularProgressLabel>
                    <VStack spacing={0}>
                      <Text fontSize="2xl" fontWeight="bold" fontFamily="mono">
                        {totalScore.toFixed(2)}
                      </Text>
                      <Text fontSize="xs" color="gray.500">raw score</Text>
                    </VStack>
                  </CircularProgressLabel>
                </CircularProgress>

                <SimpleGrid columns={2} spacing={3}>
                  {Object.entries(components).map(([key, val]) => (
                    <Stat key={key} size="sm">
                      <StatLabel fontSize="xs">{DEFAULT_COMPONENTS[key].label}</StatLabel>
                      <StatNumber fontSize="md" fontFamily="mono">{val.toFixed(3)}</StatNumber>
                      <StatHelpText fontSize="xs">w={weights[key].toFixed(2)} → {(val * weights[key]).toFixed(3)}</StatHelpText>
                    </Stat>
                  ))}
                </SimpleGrid>
              </Flex>
            </CardBody>
          </Card>

          {/* Component breakdown bars */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">Weighted Contribution Breakdown</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <VStack spacing={3} align="stretch">
                {Object.entries(DEFAULT_COMPONENTS).map(([key, meta]) => {
                  const contribution = components[key] * weights[key]
                  const pct = Math.min(100, (contribution / (totalScore || 1)) * 100)
                  return (
                    <Box key={key}>
                      <Flex justify="space-between" mb={1}>
                        <HStack spacing={2}>
                          <Badge colorScheme={meta.color} variant="subtle" fontSize="xs">{key}</Badge>
                          <Text fontSize="xs">{meta.label}</Text>
                        </HStack>
                        <Text fontSize="xs" fontFamily="mono">{contribution.toFixed(3)}</Text>
                      </Flex>
                      <Tooltip label={meta.desc} hasArrow fontSize="xs">
                        <Progress
                          value={pct}
                          colorScheme={meta.color}
                          size="sm"
                          borderRadius="full"
                        />
                      </Tooltip>
                    </Box>
                  )
                })}
              </VStack>
            </CardBody>
          </Card>

          {/* Radar chart */}
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Heading size="sm">VRE Radar Profile</Heading>
            </CardHeader>
            <CardBody pt={2}>
              <Box h="280px">
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart data={radarData}>
                    <PolarGrid stroke={useColorModeValue('#e2e8f0', '#2d3748')} />
                    <PolarAngleAxis dataKey="subject" tick={{ fontSize: 11 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9 }} />
                    <Radar
                      name="VRE Score"
                      dataKey="value"
                      stroke="#805AD5"
                      fill="#805AD5"
                      fillOpacity={0.35}
                    />
                    <ChartTooltip formatter={(v) => [`${v.toFixed(1)}%`, 'Normalized Score']} />
                  </RadarChart>
                </ResponsiveContainer>
              </Box>
            </CardBody>
          </Card>
        </VStack>
      </GridItem>
    </Grid>
  )
}
