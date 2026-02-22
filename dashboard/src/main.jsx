import React from 'react'
import ReactDOM from 'react-dom/client'
import { ChakraProvider, extendTheme, ColorModeScript } from '@chakra-ui/react'
import App from './App'

const theme = extendTheme({
  config: {
    initialColorMode: 'dark',
    useSystemColorMode: false,
  },
  colors: {
    brand: {
      50: '#f5f0ff',
      100: '#ebe0ff',
      200: '#d6c2ff',
      300: '#b899ff',
      400: '#9966ff',
      500: '#7c3aed',
      600: '#6d28d9',
      700: '#5b21b6',
      800: '#4c1d95',
      900: '#2e1065',
    },
  },
  fonts: {
    heading: `'Inter', -apple-system, BlinkMacSystemFont, sans-serif`,
    body: `'Inter', -apple-system, BlinkMacSystemFont, sans-serif`,
    mono: `'JetBrains Mono', 'Fira Code', 'Courier New', monospace`,
  },
})

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <ColorModeScript initialColorMode={theme.config.initialColorMode} />
    <ChakraProvider theme={theme}>
      <App />
    </ChakraProvider>
  </React.StrictMode>
)
