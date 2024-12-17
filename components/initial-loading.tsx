'use client'

import { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import { LoadingPage } from './loading-page'

export function InitialLoading({ children }: { children: React.ReactNode }) {
  const [isLoading, setIsLoading] = useState(true)
  const pathname = usePathname()

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false)
    }, 1500)

    return () => clearTimeout(timer)
  }, [])

  if (pathname !== '/' || !isLoading) {
    return <>{children}</>
  }

  return <LoadingPage />
}

