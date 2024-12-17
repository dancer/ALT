import Link from 'next/link'
import { Lock } from 'lucide-react'

interface ProjectCardProps {
  title: string
  category: string
  description: string
  image: string
  locked?: boolean
  slug?: string
}

export function ProjectCard({ title, category, description, image, locked, slug }: ProjectCardProps) {
  const isVideo = image.endsWith('.mp4')
  const MediaContent = () => (
    <div className="relative w-full h-full">
      {isVideo ? (
        <video
          src={image}
          autoPlay
          loop
          muted
          playsInline
          className="object-cover w-full h-full"
        />
      ) : (
        <img
          src={image}
          alt={title}
          className="object-cover w-full h-full"
        />
      )}
      <div className="absolute inset-0 bg-black opacity-30"></div>
    </div>
  )

  return (
    <div className="relative">
      <div className="aspect-video overflow-hidden bg-white relative">
        {locked && (
          <div className="absolute bottom-2 left-2 text-white z-10">
            <Lock className="w-4 h-4" />
          </div>
        )}
        {slug ? (
          <Link href={`/projects/${slug}`} className="block w-full h-full">
            <MediaContent />
          </Link>
        ) : (
          <MediaContent />
        )}
      </div>
      <div className="mt-2">
        <h3 className="text-xs uppercase">
          {slug ? (
            <Link href={`/projects/${slug}`} className="hover:text-[#C1E1C1] transition-colors duration-200 ease-in-out">
              {title}
            </Link>
          ) : (
            title
          )}
        </h3>
        <p className="text-[10px] uppercase text-neutral-600">{category}</p>
        <p className="text-sm mt-1">{description}</p>
      </div>
    </div>
  )
}

