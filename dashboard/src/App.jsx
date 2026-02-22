import {
  Box,
  Container,
  Flex,
  Heading,
  Tab,
  TabList,
  TabPanel,
  TabPanels,
  Tabs,
  Text,
  useColorMode,
  useColorModeValue,
  IconButton,
  Badge,
  VStack,
  HStack,
} from '@chakra-ui/react'
import { MoonIcon, SunIcon } from '@chakra-ui/icons'
import QuantumCircuit from './components/QuantumCircuit'
import TimeSeries from './components/TimeSeries'
import VREScore from './components/VREScore'
import HFPCalculator from './components/HFPCalculator'
import SoundWave from './components/SoundWave'

function ColorModeToggle() {
  const { colorMode, toggleColorMode } = useColorMode()
  return (
    <IconButton
      aria-label="Toggle color mode"
      icon={colorMode === 'light' ? <MoonIcon /> : <SunIcon />}
      onClick={toggleColorMode}
      variant="ghost"
      size="sm"
    />
  )
}

export default function App() {
  const bg = useColorModeValue('gray.50', 'gray.900')
  const headerBg = useColorModeValue('white', 'gray.800')
  const borderColor = useColorModeValue('gray.200', 'gray.700')

  return (
    <Box minH="100vh" bg={bg}>
      {/* Header */}
      <Box
        bg={headerBg}
        borderBottom="1px"
        borderColor={borderColor}
        px={6}
        py={3}
        position="sticky"
        top={0}
        zIndex={10}
        boxShadow="sm"
      >
        <Flex align="center" justify="space-between" maxW="1400px" mx="auto">
          <HStack spacing={3}>
            <Box
              w={8}
              h={8}
              bg="purple.500"
              borderRadius="md"
              display="flex"
              alignItems="center"
              justifyContent="center"
            >
              <Text color="white" fontWeight="bold" fontSize="sm">
                QL
              </Text>
            </Box>
            <VStack spacing={0} align="start">
              <Heading size="sm" letterSpacing="tight">
                Quantum-Logic Dashboard
              </Heading>
              <Text fontSize="xs" color="gray.500">
                UE5 Algorithm Visualizer
              </Text>
            </VStack>
            <Badge colorScheme="purple" variant="subtle" fontSize="xs">
              v1.0
            </Badge>
          </HStack>
          <ColorModeToggle />
        </Flex>
      </Box>

      {/* Main Content */}
      <Container maxW="1400px" py={6} px={4}>
        <Tabs variant="enclosed" colorScheme="purple" isLazy>
          <TabList mb={4} flexWrap="wrap" gap={1}>
            <Tab fontSize="sm" fontWeight="medium">
              Quantum Circuit
            </Tab>
            <Tab fontSize="sm" fontWeight="medium">
              Time Series / LSTM
            </Tab>
            <Tab fontSize="sm" fontWeight="medium">
              VRE Score
            </Tab>
            <Tab fontSize="sm" fontWeight="medium">
              HFP / DFA
            </Tab>
            <Tab fontSize="sm" fontWeight="medium">
              Sound Wave
            </Tab>
          </TabList>

          <TabPanels>
            <TabPanel p={0}>
              <QuantumCircuit />
            </TabPanel>
            <TabPanel p={0}>
              <TimeSeries />
            </TabPanel>
            <TabPanel p={0}>
              <VREScore />
            </TabPanel>
            <TabPanel p={0}>
              <HFPCalculator />
            </TabPanel>
            <TabPanel p={0}>
              <SoundWave />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </Container>
    </Box>
  )
}
