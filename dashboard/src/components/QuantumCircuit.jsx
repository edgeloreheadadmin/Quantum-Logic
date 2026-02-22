import { useState, useCallback } from 'react'
import {
  Box,
  Button,
  Card,
  CardBody,
  CardHeader,
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
  Select,
  SimpleGrid,
  Stat,
  StatLabel,
  StatNumber,
  Tag,
  TagLabel,
  Text,
  Tooltip,
  useColorModeValue,
  VStack,
  Badge,
  Divider,
  Alert,
  AlertIcon,
  AlertDescription,
} from '@chakra-ui/react'

// --- Quantum simulation logic (JS port of UE5_scriptsv2.cpp) ---

function createQubit() {
  return { state0: 1.0, state1: 0.0 } // |0> state
}

function applyHadamard(qubit) {
  const s = 1 / Math.sqrt(2)
  return {
    state0: (qubit.state0 + qubit.state1) * s,
    state1: (qubit.state0 - qubit.state1) * s,
  }
}

function measureQubit(qubit) {
  const prob0 = qubit.state0 * qubit.state0
  return Math.random() < prob0 ? 0 : 1
}

function applyCNOT(qubits, control, target) {
  // Conceptual CNOT: if control qubit measured as |1>, flip target
  const controlState = qubits[control].state1 * qubits[control].state1 > 0.5 ? 1 : 0
  if (controlState === 1) {
    return qubits.map((q, i) =>
      i === target ? { state0: q.state1, state1: q.state0 } : q
    )
  }
  return qubits
}

function initCircuit(numQubits) {
  return Array.from({ length: numQubits }, createQubit)
}

// Gate history entry: { type: 'H'|'CNOT', qubit, control?, target? }

const GATE_COLORS = {
  H: 'purple',
  CNOT: 'blue',
  X: 'red',
  Z: 'green',
}

function GateButton({ label, colorScheme, onClick, tooltip }) {
  return (
    <Tooltip label={tooltip} hasArrow>
      <Button size="sm" colorScheme={colorScheme} onClick={onClick} fontFamily="mono" fontWeight="bold">
        {label}
      </Button>
    </Tooltip>
  )
}

function QubitRow({ index, qubit, measurement, gateLog }) {
  const bg = useColorModeValue('gray.50', 'gray.750')
  const borderColor = useColorModeValue('gray.200', 'gray.600')
  const prob1 = qubit.state1 * qubit.state1
  const prob0 = qubit.state0 * qubit.state0

  return (
    <Box border="1px" borderColor={borderColor} borderRadius="lg" p={3} bg={bg}>
      <Flex align="center" gap={4} flexWrap="wrap">
        {/* Label */}
        <Box minW="32px">
          <Text fontFamily="mono" fontWeight="bold" fontSize="sm" color="purple.400">
            |q{index}⟩
          </Text>
        </Box>

        {/* State bar */}
        <Box flex={1} minW="120px">
          <Flex gap={1} mb={1}>
            <Text fontSize="xs" color="gray.500">|0⟩ {(prob0 * 100).toFixed(1)}%</Text>
            <Text fontSize="xs" color="gray.500" ml="auto">|1⟩ {(prob1 * 100).toFixed(1)}%</Text>
          </Flex>
          <Box w="full" h="6px" borderRadius="full" bg="gray.200" overflow="hidden">
            <Box
              h="full"
              borderRadius="full"
              bg="purple.400"
              w={`${prob0 * 100}%`}
              transition="width 0.3s"
            />
          </Box>
        </Box>

        {/* Amplitudes */}
        <HStack spacing={2}>
          <Badge colorScheme="purple" fontFamily="mono" fontSize="xs">
            α={qubit.state0.toFixed(3)}
          </Badge>
          <Badge colorScheme="pink" fontFamily="mono" fontSize="xs">
            β={qubit.state1.toFixed(3)}
          </Badge>
        </HStack>

        {/* Measurement result */}
        {measurement !== null && (
          <Badge
            colorScheme={measurement === 0 ? 'green' : 'orange'}
            variant="solid"
            fontSize="sm"
            px={2}
          >
            → |{measurement}⟩
          </Badge>
        )}
      </Flex>

      {/* Applied gates */}
      {gateLog.length > 0 && (
        <Flex mt={2} gap={1} flexWrap="wrap">
          {gateLog.map((g, i) => (
            <Tag key={i} size="sm" colorScheme={GATE_COLORS[g.type] || 'gray'} variant="subtle">
              <TagLabel fontFamily="mono" fontSize="xs">{g.label}</TagLabel>
            </Tag>
          ))}
        </Flex>
      )}
    </Box>
  )
}

export default function QuantumCircuit() {
  const [numQubits, setNumQubits] = useState(3)
  const [qubits, setQubits] = useState(() => initCircuit(3))
  const [measurements, setMeasurements] = useState([])
  const [gateHistory, setGateHistory] = useState([]) // [{qubitIdx, label, type}]
  const [cnot, setCnot] = useState({ control: '0', target: '1' })
  const [selectedQubit, setSelectedQubit] = useState('0')

  const cardBg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.600')

  const reset = useCallback((n = numQubits) => {
    setQubits(initCircuit(n))
    setMeasurements([])
    setGateHistory([])
  }, [numQubits])

  const handleNumQubitsChange = (val) => {
    const n = parseInt(val, 10)
    setNumQubits(n)
    reset(n)
  }

  const applyH = () => {
    const qi = parseInt(selectedQubit, 10)
    setQubits(prev => prev.map((q, i) => i === qi ? applyHadamard(q) : q))
    setGateHistory(prev => [...prev, { qubitIdx: qi, label: 'H', type: 'H' }])
    setMeasurements([])
  }

  const applyX = () => {
    const qi = parseInt(selectedQubit, 10)
    setQubits(prev =>
      prev.map((q, i) => i === qi ? { state0: q.state1, state1: q.state0 } : q)
    )
    setGateHistory(prev => [...prev, { qubitIdx: qi, label: 'X', type: 'X' }])
    setMeasurements([])
  }

  const applyZ = () => {
    const qi = parseInt(selectedQubit, 10)
    setQubits(prev =>
      prev.map((q, i) => i === qi ? { state0: q.state0, state1: -q.state1 } : q)
    )
    setGateHistory(prev => [...prev, { qubitIdx: qi, label: 'Z', type: 'Z' }])
    setMeasurements([])
  }

  const applyCnotGate = () => {
    const ctrl = parseInt(cnot.control, 10)
    const tgt = parseInt(cnot.target, 10)
    if (ctrl === tgt) return
    setQubits(prev => applyCNOT(prev, ctrl, tgt))
    setGateHistory(prev => [
      ...prev,
      { qubitIdx: ctrl, label: `CNOT(c=${ctrl})`, type: 'CNOT' },
      { qubitIdx: tgt, label: `CNOT(t=${tgt})`, type: 'CNOT' },
    ])
    setMeasurements([])
  }

  const measureAll = () => {
    const results = qubits.map(q => measureQubit(q))
    setMeasurements(results)
    // Collapse qubits
    setQubits(prev =>
      prev.map((q, i) =>
        results[i] === 0 ? { state0: 1, state1: 0 } : { state0: 0, state1: 1 }
      )
    )
  }

  const qubitOptions = Array.from({ length: numQubits }, (_, i) => i)

  // Per-qubit gate log
  const perQubitGates = Array.from({ length: numQubits }, (_, qi) =>
    gateHistory.filter(g => g.qubitIdx === qi)
  )

  // Decoded classical value from measurements
  const classicalValue =
    measurements.length > 0
      ? measurements.reduce((acc, bit, i) => acc | (bit << (measurements.length - 1 - i)), 0)
      : null

  return (
    <Grid templateColumns={{ base: '1fr', lg: '320px 1fr' }} gap={4}>
      {/* Controls */}
      <GridItem>
        <Card bg={cardBg} border="1px" borderColor={borderColor}>
          <CardHeader pb={2}>
            <Heading size="sm">Circuit Controls</Heading>
          </CardHeader>
          <CardBody pt={2}>
            <VStack spacing={4} align="stretch">
              <FormControl>
                <FormLabel fontSize="xs" mb={1}>Number of Qubits</FormLabel>
                <NumberInput
                  min={1}
                  max={8}
                  value={numQubits}
                  onChange={handleNumQubitsChange}
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

              <FormControl>
                <FormLabel fontSize="xs" mb={1}>Target Qubit</FormLabel>
                <Select
                  size="sm"
                  value={selectedQubit}
                  onChange={e => setSelectedQubit(e.target.value)}
                >
                  {qubitOptions.map(i => (
                    <option key={i} value={i}>q{i}</option>
                  ))}
                </Select>
              </FormControl>

              <SimpleGrid columns={3} spacing={2}>
                <GateButton
                  label="H"
                  colorScheme="purple"
                  onClick={applyH}
                  tooltip="Hadamard gate: puts qubit into superposition"
                />
                <GateButton
                  label="X"
                  colorScheme="red"
                  onClick={applyX}
                  tooltip="Pauli-X (NOT) gate: flips |0⟩↔|1⟩"
                />
                <GateButton
                  label="Z"
                  colorScheme="green"
                  onClick={applyZ}
                  tooltip="Pauli-Z gate: flips phase of |1⟩"
                />
              </SimpleGrid>

              <Divider />

              <Box>
                <Text fontSize="xs" fontWeight="semibold" mb={2} color="gray.500" textTransform="uppercase">
                  CNOT Gate
                </Text>
                <SimpleGrid columns={2} spacing={2} mb={2}>
                  <FormControl>
                    <FormLabel fontSize="xs" mb={1}>Control</FormLabel>
                    <Select
                      size="sm"
                      value={cnot.control}
                      onChange={e => setCnot(prev => ({ ...prev, control: e.target.value }))}
                    >
                      {qubitOptions.map(i => (
                        <option key={i} value={i}>q{i}</option>
                      ))}
                    </Select>
                  </FormControl>
                  <FormControl>
                    <FormLabel fontSize="xs" mb={1}>Target</FormLabel>
                    <Select
                      size="sm"
                      value={cnot.target}
                      onChange={e => setCnot(prev => ({ ...prev, target: e.target.value }))}
                    >
                      {qubitOptions.map(i => (
                        <option key={i} value={i}>q{i}</option>
                      ))}
                    </Select>
                  </FormControl>
                </SimpleGrid>
                <Button size="sm" colorScheme="blue" w="full" onClick={applyCnotGate}>
                  Apply CNOT
                </Button>
              </Box>

              <Divider />

              <Button colorScheme="teal" onClick={measureAll} size="sm">
                Measure All Qubits
              </Button>
              <Button variant="outline" size="sm" onClick={() => reset()}>
                Reset Circuit
              </Button>
            </VStack>
          </CardBody>
        </Card>
      </GridItem>

      {/* Circuit visualization */}
      <GridItem>
        <VStack spacing={3} align="stretch">
          <Card bg={cardBg} border="1px" borderColor={borderColor}>
            <CardHeader pb={2}>
              <Flex justify="space-between" align="center">
                <Heading size="sm">Qubit State Register</Heading>
                <Badge colorScheme="purple" variant="outline">
                  {numQubits} qubit{numQubits !== 1 ? 's' : ''}
                </Badge>
              </Flex>
            </CardHeader>
            <CardBody pt={2}>
              <VStack spacing={2} align="stretch">
                {qubits.map((q, i) => (
                  <QubitRow
                    key={i}
                    index={i}
                    qubit={q}
                    measurement={measurements[i] ?? null}
                    gateLog={perQubitGates[i]}
                  />
                ))}
              </VStack>
            </CardBody>
          </Card>

          {/* Measurement results */}
          {measurements.length > 0 && (
            <Card bg={cardBg} border="1px" borderColor={borderColor}>
              <CardHeader pb={2}>
                <Heading size="sm">Measurement Results</Heading>
              </CardHeader>
              <CardBody pt={2}>
                <SimpleGrid columns={{ base: 2, md: 4 }} spacing={3}>
                  <Stat>
                    <StatLabel fontSize="xs">Bitstring</StatLabel>
                    <StatNumber fontSize="lg" fontFamily="mono">
                      {measurements.join('')}
                    </StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel fontSize="xs">Classical Value</StatLabel>
                    <StatNumber fontSize="lg" fontFamily="mono">
                      {classicalValue}
                    </StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel fontSize="xs">Qubits in |1⟩</StatLabel>
                    <StatNumber fontSize="lg">
                      {measurements.filter(m => m === 1).length}
                    </StatNumber>
                  </Stat>
                  <Stat>
                    <StatLabel fontSize="xs">Gates Applied</StatLabel>
                    <StatNumber fontSize="lg">
                      {gateHistory.length}
                    </StatNumber>
                  </Stat>
                </SimpleGrid>
                <Alert status="info" mt={3} borderRadius="md" size="sm">
                  <AlertIcon />
                  <AlertDescription fontSize="xs">
                    After measurement the circuit collapsed to the classical state |{measurements.join('')}⟩
                    = decimal {classicalValue}.
                  </AlertDescription>
                </Alert>
              </CardBody>
            </Card>
          )}
        </VStack>
      </GridItem>
    </Grid>
  )
}
