import { useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'

/** Plays forward to the end, then plays back in reverse to the start, then
 * forward again — a ping-pong loop instead of the jump-cut restart a plain
 * `loop` attribute gives you. <video> doesn't support negative playbackRate
 * in any mainstream browser, so the reverse pass is driven by hand: pause
 * native playback and step `currentTime` backwards every animation frame. */
function usePingPongVideo(videoRef) {
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    let rafId = null
    let lastTs = null

    const stepReverse = (ts) => {
      if (lastTs == null) lastTs = ts
      const dt = (ts - lastTs) / 1000
      lastTs = ts

      const next = video.currentTime - dt
      if (next <= 0) {
        video.currentTime = 0
        lastTs = null
        video.play().catch(() => {})
        return
      }
      video.currentTime = next
      rafId = requestAnimationFrame(stepReverse)
    }

    const handleEnded = () => {
      lastTs = null
      rafId = requestAnimationFrame(stepReverse)
    }

    video.addEventListener('ended', handleEnded)
    video.play().catch(() => {})

    return () => {
      video.removeEventListener('ended', handleEnded)
      if (rafId) cancelAnimationFrame(rafId)
    }
  }, [videoRef])
}

export default function Landing() {
  const navigate = useNavigate()
  const videoRef = useRef(null)
  usePingPongVideo(videoRef)

  return (
    <div className="relative w-full min-h-screen overflow-hidden bg-[#0a0d0b]">
      <video
        ref={videoRef}
        className="absolute inset-0 w-full h-full object-cover"
        src="/hero.mp4" muted playsInline
      />
      <div className="absolute inset-0 bg-black/55" />
      <div className="absolute inset-0 bg-gradient-to-b from-black/35 via-black/20 to-black/75" />
      <div
        className="absolute -top-[10%] left-1/2 -translate-x-1/2 w-[1000px] h-[720px] pointer-events-none"
        style={{ background: 'radial-gradient(ellipse at 50% 30%, rgba(10,132,255,0.22) 0%, transparent 68%)' }}
      />

      {/* Navbar — floating glass pill, consistent with the app shell's top nav */}
      <motion.nav
        initial={{ opacity: 0, y: -16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: 'easeOut' }}
        className="relative z-10 mx-3 sm:mx-5 mt-4"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between gap-4 px-4 sm:px-5 py-2.5 rounded-[28px] bg-white/[0.06] border border-white/15 backdrop-blur-2xl shadow-xl shadow-black/10">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-full bg-signal flex items-center justify-center text-white font-display text-sm shadow-sm shadow-signal/40">
              D
            </div>
            <span className="font-display text-[15px] text-paper">DocuMind</span>
            <span className="hidden sm:inline font-mono text-[11px] text-paper/55 ml-1">evidence-first documents</span>
          </div>
          <motion.button
            whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
            onClick={() => navigate('/login')}
            className="px-5 py-2.5 rounded-full text-sm font-semibold text-ink bg-paper shadow-lg"
          >
            Log in
          </motion.button>
        </div>
      </motion.nav>

      {/* Hero content */}
      <div className="relative z-10 flex flex-col items-start justify-center min-h-[calc(100vh-88px)] px-6 md:px-11 max-w-2xl">
        <motion.div
          initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.1, ease: 'easeOut' }}
          className="inline-flex items-center gap-2.5 rounded-full pl-2 pr-4 py-1.5 bg-white/[0.06] border border-white/15 backdrop-blur-sm"
        >
          <div className="flex">
            {[0, 1, 2].map((i) => (
              <div key={i} className={`w-[22px] h-[22px] rounded-full bg-gradient-to-br from-signal to-signalDim border-2 border-[#0a0d0b] ${i === 0 ? '' : '-ml-2'}`} />
            ))}
          </div>
          <span className="text-xs font-mono text-paper/75">
            Grounded answers, <strong className="text-paper font-semibold">not guesses</strong>
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 22 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.25, ease: 'easeOut' }}
          className="font-display font-medium text-paper mt-6 leading-[1.08] tracking-tight"
          style={{ fontSize: 'clamp(2.4rem, 5.2vw, 4rem)' }}
        >
          Turn documents<br />into knowledge.
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.42, ease: 'easeOut' }}
          className="mt-5 text-[15px] leading-relaxed text-paper/60 max-w-md"
        >
          Ask questions and get answers traced back to the exact page and passage
          they came from — with an honest "not found" instead of a confident guess.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.8, delay: 0.58, ease: 'easeOut' }}
          className="flex items-center gap-3.5 mt-9"
        >
          <motion.button
            whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
            onClick={() => navigate('/login?mode=register')}
            className="px-7 py-3.5 rounded-full bg-ink text-paper text-sm font-semibold border border-white/15"
          >
            Get Started
          </motion.button>
          <motion.button
            whileHover={{ scale: 1.03 }} whileTap={{ scale: 0.97 }}
            onClick={() => navigate('/login')}
            className="px-7 py-3.5 rounded-full bg-white/10 text-paper text-sm font-semibold border border-white/20 backdrop-blur-sm"
          >
            Log in
          </motion.button>
        </motion.div>
      </div>
    </div>
  )
}
