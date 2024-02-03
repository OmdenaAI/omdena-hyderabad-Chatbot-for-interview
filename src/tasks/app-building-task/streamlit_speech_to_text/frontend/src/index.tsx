import React from "react"
import ReactDOM from "react-dom"

// Lots of import to define a Styletron engine and load the light theme of baseui
import { DarkTheme, ThemeProvider } from "baseui"
import { Client as Styletron } from "styletron-engine-atomic"
import { Provider as StyletronProvider } from "styletron-react"
import SpeechToText from "./SpeechToText"

const engine = new Styletron()

// Wrap your CustomSlider with the baseui them
ReactDOM.render(
  <React.StrictMode>
    <StyletronProvider value={engine}>
      <ThemeProvider theme={DarkTheme}>
        <SpeechToText />
      </ThemeProvider>
    </StyletronProvider>
  </React.StrictMode>,
  document.getElementById("root")
)
