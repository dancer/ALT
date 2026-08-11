'use client'

import { GrainGradient } from '@paper-design/shaders-react'

export function GradientBackground() {
  return (
    <div className="backdrop" aria-hidden="true">
      <GrainGradient
        style={{ height: '100%', width: '100%' }}
        colorBack="hsl(0, 0%, 0%)"
        softness={0.76}
        intensity={0.45}
        noise={0}
        shape="corners"
        offsetX={0}
        offsetY={0}
        scale={1}
        rotation={0}
        speed={1}
        colors={['hsl(28, 90%, 62%)', 'hsl(36, 100%, 78%)', 'hsl(18, 100%, 52%)']}
      />
      <div className="backdrop-shade" />
    </div>
  )
}
