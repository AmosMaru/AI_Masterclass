import { useRef, useState, useEffect } from "react";
import { useChat } from "../hooks/useChat";

export const UI = ({ hidden, ...props }) => {
  const input = useRef();
  const { chat, loading, cameraZoomed, setCameraZoomed, message } = useChat();
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [recognition, setRecognition] = useState(null);

  // Initialize speech recognition on component mount
  useEffect(() => {
    if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
      const SpeechRecognition =
        window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognitionInstance = new SpeechRecognition();

      recognitionInstance.continuous = false;
      recognitionInstance.interimResults = true;
      recognitionInstance.lang = "en-US";

      recognitionInstance.onstart = () => {
        setIsListening(true);
        setTranscript("");
      };

      recognitionInstance.onresult = (event) => {
        const current = event.resultIndex;
        const result = event.results[current][0].transcript;
        setTranscript(result);

        if (input.current) {
          input.current.value = result;
        }

        // If this is a final result, we'll know by checking isFinal property
        if (event.results[current].isFinal) {
          // Wait a short moment before sending to ensure recognition is complete
          setTimeout(() => {
            if (result.trim()) {
              sendMessage(result);
              recognitionInstance.stop();
            }
          }, 500);
        }
      };

      recognitionInstance.onend = () => {
        setIsListening(false);
      };

      recognitionInstance.onerror = (event) => {
        console.error("Speech recognition error", event.error);
        setIsListening(false);
      };

      setRecognition(recognitionInstance);
    }

    // Cleanup on unmount
    return () => {
      if (recognition) {
        recognition.stop();
      }
    };
  }, []);

  const toggleListening = () => {
    if (!recognition) return;

    if (isListening) {
      recognition.stop();
    } else {
      try {
        recognition.start();
      } catch (error) {
        // Handle cases where recognition is already running
        console.log("Recognition already started or other error:", error);
      }
    }
  };

  const sendMessage = (text = null) => {
    const messageText = text || input.current.value;
    if (!loading && !message && messageText.trim()) {
      chat(messageText);
      input.current.value = "";
      setTranscript("");
    }
  };

  if (hidden) {
    return null;
  }

  return (
    <>
      <div className="fixed top-0 left-0 right-0 bottom-0 z-10 flex justify-between p-4 flex-col pointer-events-none">
        <div className="self-start backdrop-blur-md bg-white bg-opacity-50 p-4 rounded-lg">
          <h1 className="font-black text-xl">Powered By: </h1>
            <img src="./image.png" alt="placeholder" className="mt-2 rounded-md w-30 h-20"  />
        </div>
        <div className="w-full flex flex-col items-end justify-center gap-4">
          <button
            onClick={() => setCameraZoomed(!cameraZoomed)}
            className="pointer-events-auto bg-pink-500 hover:bg-pink-600 text-white p-4 rounded-md"
          >
            {cameraZoomed ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM13.5 10.5h-6"
                />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607zM10.5 7.5v6m3-3h-6"
                />
              </svg>
            )}
          </button>
        </div>
        <div className="flex items-center gap-2 pointer-events-auto max-w-screen-sm w-full mx-auto">
          <div className="relative w-full">
            <input
              className="w-full placeholder:text-gray-800 placeholder:italic p-4 rounded-md bg-opacity-50 bg-white backdrop-blur-md pr-12"
              placeholder={isListening ? "Listening..." : "Type a message..."}
              ref={input}
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  sendMessage();
                }
              }}
            />
            <button
              onClick={toggleListening}
              className={`absolute right-2 top-1/2 transform -translate-y-1/2 p-2 rounded-full ${
                isListening
                  ? "bg-red-500 text-white animate-pulse"
                  : "bg-gray-200 text-gray-700 hover:bg-gray-300"
              }`}
              title={isListening ? "Stop listening" : "Start listening"}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-5 h-5"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z"
                />
              </svg>
            </button>
          </div>
          <button
            disabled={loading || message || !transcript.trim()}
            onClick={() => sendMessage()}
            className={`bg-pink-500 hover:bg-pink-600 text-white p-4 px-10 font-semibold uppercase rounded-md ${
              loading || message || !transcript.trim()
                ? "cursor-not-allowed opacity-30"
                : ""
            }`}
          >
            Send
          </button>
        </div>

        {isListening && (
          <div className="pointer-events-auto mx-auto mt-2 px-4 py-2 bg-white bg-opacity-75 backdrop-blur-sm rounded-full text-sm text-gray-700">
            Speak now... message will send automatically when you finish
          </div>
        )}
      </div>
    </>
  );
};
