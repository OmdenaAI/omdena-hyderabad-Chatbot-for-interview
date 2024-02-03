//@ts-nocheck
import React, { useEffect } from "react"
import SpeechRecognition, {
  useSpeechRecognition,
} from "react-speech-recognition"
import {
  ComponentProps,
  Streamlit,
  withStreamlitConnection,
} from "streamlit-component-lib"

import { useState } from "react"

/**
 * We can use a Typescript interface to destructure the arguments from Python
 * and validate the types of the input
 */
// interface PythonArgs {
//   label: string
//   minValue?: number
//   maxValue?: number
//   initialValue: number[]
// }

/**
 * No more props manipulation in the code.
 * We store props in state and pass value directly to underlying Slider
 * and then back to Streamlit.
 */
const SpeechToText = (props: ComponentProps) => {
  const { transcript, resetTranscript } = useSpeechRecognition()
  const [isListening, setIsListening] = useState(false)

  const handleListing = () => {
    setIsListening(true)
    SpeechRecognition.startListening({
      continuous: true,
    })
  }
  const stopHandle = () => {
    setIsListening(false)
    SpeechRecognition.stopListening()
    Streamlit.setComponentValue({ transcript, stopped: true })
    resetTranscript()
  }

  useEffect(() => {
    Streamlit.setFrameHeight()
    console.log({ transcript })
    if (transcript !== "") Streamlit.setComponentValue(transcript)
  }, [transcript])
  return (
    <div className="microphone-wrapper">
      <div className="mircophone-container">
        {!isListening ? (
          <div
            className="microphone-icon-container btn btn-sm btn-outline-success"
            onClick={handleListing}
          >
            Start Recording
          </div>
        ) : (
          <button
            className="microphone-stop btn btn-sm btn-outline-danger"
            onClick={stopHandle}
          >
            Stop Recording
          </button>
        )}
      </div>
    </div>
  )
}

export default withStreamlitConnection(SpeechToText)
