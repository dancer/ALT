import { Header } from '@/components/header'
import { TopSection } from '@/components/top-section'
import { ProjectCard } from '@/components/project-card'

export default function Home() {
  return (
    <>
      <Header />
      
      <main className="pt-20 main-page">
        <div className="max-w-[2000px] mx-auto px-6">
          <TopSection />

          <section>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <ProjectCard
                title="GNAR.AI"
                category="Coming Soon"
                description="Innovative AI project under development"
                image="/gnar.mp4"
                locked
              />
            <ProjectCard
                title="Nidal.ee"
                category="Gaming"
                description="Streamlined account manager for Riot Games launchers"
                image="/nidalee.png"
                slug="nidalee"
              />
            <ProjectCard
                title="Tre.ad"
                category="Developer Tools"
                description="Coding speed test platform for measuring and improving developer velocity"
                image="/tread.png"
                slug="tread"
              />
              <ProjectCard
                title="Spiritu.al"
                category="AI Tools"
                description="Advanced prompt engineering platform enhancing AI interactions"
                image="/spiritual.png"
                slug="spiritual"
              />
              <ProjectCard
                title="Lovehea.rt"
                category="Fintech"
                description="Innovative stablecoin payment platform with empathy-driven features"
                image="/loveheart.png"
                slug="loveheart"
              />
              <ProjectCard
                title="Line.al"
                category="UI Library"
                description="Comprehensive Terminal UI component library for command-line interfaces"
                image="/lineal.png"
                slug="lineal"
              />
              <ProjectCard
                title="Pictori.al"
                category="Social"
                description="Visual harmony-based profile picture matching platform"
                image="/pictorial.png"
                slug="pictorial"
              />
              <ProjectCard
                title="Hecar.im"
                category="Developer Tools"
                description="Customized VS Code theme for enhanced coding experience"
                image="/hecarim.png"
                slug="hecarim"
              />
              <ProjectCard
                title="Reng.ar"
                category="Coming Soon"
                description="Next-generation project in development"
                image="/rengar.png"
                locked
              />
            </div>
          </section>
        </div>
      </main>
    </>
  )
}

