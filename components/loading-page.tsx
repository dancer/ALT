'use client'

import { useState, useEffect } from 'react'

export function LoadingPage() {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval)
          return 100
        }
        return prev + 1
      })
    }, 15) // Changed from 25 to 15 (1500ms / 100 steps = 15ms per step)

    return () => clearInterval(interval)
  }, [])

  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-[#f5f5f0] transition-opacity duration-500">
      <div className="space-y-8 text-center">
        <h1 className="text-2xl tracking-[0.3em] text-neutral-800">Anywho</h1>
        <div className="text-sm tracking-wider text-neutral-600">{progress}%</div>
      </div>
    </div>
  )
}

