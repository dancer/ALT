"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import {
  ChevronUp,
  Clock,
  Share2,
  Check,
  Brain,
  Zap,
  Sparkles,
} from "lucide-react";

export default function DiffusionModels() {
  const [scrollProgress, setScrollProgress] = useState(0);
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      const totalHeight =
        document.documentElement.scrollHeight -
        document.documentElement.clientHeight;
      const progress = (window.scrollY / totalHeight) * 100;
      setScrollProgress(progress);
      setShowScrollTop(window.scrollY > 300);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const handleShare = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("Failed to copy: ", err);
    }
  };

  return (
    <article className="relative space-y-6 text-xs">
      <div className="fixed top-0 left-0 w-full h-1 bg-gray-200">
        <div
          className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
          style={{ width: `${scrollProgress}%` }}
        ></div>
      </div>

      <Link
        href="/"
        className="text-blue-400 hover:underline inline-block mb-4"
      >
        {"<"} Back to home (in case your brain explodes)
      </Link>

      <header className="space-y-2">
        <h1 className="text-lg font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500">
          Holy Smokes, It's Diffusion Models! 🤯
        </h1>
        <p className="text-gray-400">
          Buckle up, folks! We're diving into the wild world of AI that turns
          noise into art. No, seriously.
        </p>
        <div className="flex items-center space-x-4 text-gray-500">
          <span className="flex items-center">
            <Clock size={12} className="mr-1" /> 25 min read (or 2 hours if
            you're me)
          </span>
          <button
            onClick={handleShare}
            className="flex items-center hover:text-gray-300 transition-colors"
          >
            {copied ? (
              <Check size={12} className="mr-1" />
            ) : (
              <Share2 size={12} className="mr-1" />
            )}
            {copied ? "Copied! You're awesome!" : "Share the madness"}
          </button>
        </div>
      </header>

      <nav className="border border-gray-800 rounded p-4">
        <h2 className="font-semibold mb-2">What's Coming (Brace Yourself)</h2>
        <ul className="space-y-1">
          <li>
            <a href="#intro" className="hover:text-blue-400 transition-colors">
              1. Oh God, What Have I Gotten Myself Into?
            </a>
          </li>
          <li>
            <a
              href="#foundations"
              className="hover:text-blue-400 transition-colors"
            >
              2. The "Duh" Moment: Generative AI Basics
            </a>
          </li>
          <li>
            <a
              href="#old-school"
              className="hover:text-blue-400 transition-colors"
            >
              3. The OGs: GANs and VAEs (They Walked So Diffusion Could Run)
            </a>
          </li>
          <li>
            <a
              href="#problems"
              className="hover:text-blue-400 transition-colors"
            >
              4. When Good AIs Go Bad: The Struggles
            </a>
          </li>
          <li>
            <a
              href="#diffusion-magic"
              className="hover:text-blue-400 transition-colors"
            >
              5. Enter Diffusion: The "Hold My Beer" of AI
            </a>
          </li>
          <li>
            <a
              href="#under-the-hood"
              className="hover:text-blue-400 transition-colors"
            >
              6. Under the Hood: How Does This Sorcery Work?
            </a>
          </li>
          <li>
            <a
              href="#lets-build"
              className="hover:text-blue-400 transition-colors"
            >
              7. Let's Build This Thing! (What Could Go Wrong?)
            </a>
          </li>
          <li>
            <a
              href="#mind-blown"
              className="hover:text-blue-400 transition-colors"
            >
              8. Mind = Blown (The "Now What?" Moment)
            </a>
          </li>
        </ul>
      </nav>

      <section id="intro" className="space-y-4">
        <h2 className="text-sm font-semibold flex items-center">
          <Brain className="mr-2" />
          Oh God, What Have I Gotten Myself Into?
        </h2>
        <p>
          Alright, folks, gather 'round! Today, we're diving headfirst into the
          wild world of diffusion models. And let me tell you, when I first
          started learning about this stuff, my brain felt like it was doing
          backflips while juggling flaming chainsaws. Fun times!
        </p>
        <p>
          So, why am I putting myself (and you) through this mental gymnastics?
          Because diffusion models are the cool new kid on the AI block, and
          they're doing some seriously mind-bending stuff. We're talking about
          turning random noise into masterpieces. It's like watching a digital
          Jackson Pollock transform into a Rembrandt. Magic? Nope, just math.
          Lots and lots of math.
        </p>
        <p>
          But hey, if I can wrap my head around this, so can you! So grab your
          favorite caffeinated beverage, tell your brain to sit down and pay
          attention, and let's embark on this journey of confusion,
          enlightenment, and probably more confusion. Ready? Let's go!
        </p>
      </section>

      <section id="foundations" className="space-y-4">
        <h2 className="text-sm font-semibold flex items-center">
          <Zap className="mr-2" />
          The "Duh" Moment: Generative AI Basics
        </h2>
        <p>
          Okay, before we dive into the deep end, let's paddle in the kiddie
          pool for a bit. Generative AI is all about teaching machines to create
          stuff. And when I say "stuff," I mean anything from cat pictures to
          Shakespeare sonnets. It's like giving a computer an imagination, minus
          the existential crises.
        </p>
        <p>Here's the gist:</p>
        <ul className="list-disc list-inside space-y-2">
          <li>Data distributions: Fancy way of saying "patterns in stuff"</li>
          <li>
            Latent spaces: Where AI dreams are born (cue the "Inception"
            BWAAAAH)
          </li>
          <li>
            The creativity conundrum: Make it new, but not too new. AI's got
            trust issues.
          </li>
        </ul>
        <p>
          Got it? No? Perfect! You're right where you need to be. Let's keep
          going!
        </p>
      </section>

      <section id="old-school" className="space-y-4">
        <h2 className="text-sm font-semibold flex items-center">
          <Sparkles className="mr-2" />
          The OGs: GANs and VAEs (They Walked So Diffusion Could Run)
        </h2>
        <p>
          Alright, time for a history lesson. Before diffusion models crashed
          the party, we had two big shots in the generative AI world: GANs and
          VAEs. Think of them as the cool aunts and uncles of the AI family.
        </p>
        <h3 className="text-xs font-semibold mt-2">
          GANs (Generative Adversarial Networks)
        </h3>
        <p>Imagine two AIs walking into a bar:</p>
        <ul className="list-disc list-inside space-y-2">
          <li>
            The Generator: "I bet I can create a fake ID that'll fool you!"
          </li>
          <li>The Discriminator: "Oh yeah? Bring it on, pixel-pusher!"</li>
        </ul>
        <p>
          And that, my friends, is GANs in a nutshell. Two neural networks
          duking it out until one can create forgeries good enough to fool the
          other. It's like an arms race, but with less "pew pew" and more "1s
          and 0s".
        </p>
        <h3 className="text-xs font-semibold mt-2">
          VAEs (Variational Autoencoders)
        </h3>
        <p>Now, VAEs are the more introspective cousin:</p>
        <ul className="list-disc list-inside space-y-2">
          <li>Step 1: Squish the data (like, really squish it)</li>
          <li>Step 2: Unsquish it and hope it looks right</li>
          <li>Step 3: ??? </li>
          <li>Step 4: Profit! (Or at least, generate some blurry images)</li>
        </ul>
        <p>
          VAEs are all about finding the essence of data, then recreating it.
          It's like if you described a cat to an alien, and they tried to draw
          it. Sometimes it works... other times you get a furry blob with
          whiskers.
        </p>
      </section>

      <section id="problems" className="space-y-4">
        <h2 className="text-sm font-semibold flex items-center">
          <Zap className="mr-2" />
          When Good AIs Go Bad: The Struggles
        </h2>
        <p>
          Now, you might be thinking, "These GANs and VAEs sound pretty cool!
          Why aren't we using them for everything?" Oh, sweet summer child. Let
          me introduce you to the joys of AI growing pains.
        </p>
        <h3 className="text-xs font-semibold mt-2">GAN Troubles</h3>
        <ul className="list-disc list-inside space-y-2">
          <li>Mode collapse: When your AI becomes a one-hit wonder</li>
          <li>
            Training instability: Like trying to balance a pencil on its tip...
            while riding a unicycle
          </li>
          <li>
            The "Are we there yet?" problem: How do you know when your fake
            images are fake enough?
          </li>
        </ul>
        <h3 className="text-xs font-semibold mt-2">VAE Vexations</h3>
        <ul className="list-disc list-inside space-y-2">
          <li>The blurry curse: When your AI needs glasses</li>
          <li>
            Latent space woes: "You can be anything you want!" "Cool, I want to
            be a potato."
          </li>
          <li>
            Identity crisis: Trying to be good at recreating AND generating.
            Talk about pressure!
          </li>
        </ul>
        <p>
          These issues had AI researchers tearing their hair out (or would have,
          if they hadn't already lost it trying to debug their code). But fear
          not! Our knight in shining armor is about to enter the scene...
        </p>
      </section>

      <section id="diffusion-magic" className="space-y-4">
        <h2 className="text-sm font-semibold flex items-center">
          <Sparkles className="mr-2" />
          Enter Diffusion: The "Hold My Beer" of AI
        </h2>
        <p>
          Just when everyone thought generative AI couldn't get any weirder,
          diffusion models said, "Challenge accepted!" These models took one
          look at the existing problems and decided, "You know what would fix
          this? MORE NOISE!"
        </p>
        <p>
          Here's the mind-bending part: Diffusion models learn by destroying
          information, then figuring out how to recreate it. It's like if you
          learned to bake a cake by watching someone un-bake it, ingredient by
          ingredient. Sounds crazy? Welcome to the club!
        </p>
        <h3 className="text-xs font-semibold mt-2">The Diffusion Dance</h3>
        <ol className="list-decimal list-inside space-y-2">
          <li>Start with a nice, clean image</li>
          <li>
            Add noise. More noise. No, even more. Keep going until it looks like
            TV static
          </li>
          <li>Now, try to undo that mess, one step at a time</li>
          <li>???</li>
          <li>Profit! (But this time, with sharp, diverse images)</li>
        </ol>
        <p>
          If you're scratching your head right now, congratulations! You're
          starting to understand diffusion models. Or you have dandruff. Either
          way, let's keep going!
        </p>
      </section>

      <section id="under-the-hood" className="space-y-4">
        <h2 className="text-sm font-semibold flex items-center">
          <Brain className="mr-2" />
          Under the Hood: How Does This Sorcery Work?
        </h2>
        <p>
          Alright, brace yourselves. We're about to pop the hood on this AI
          engine and peek at the math. Don't worry, I'll try to keep the
          equations to a minimum. No promises about the headaches, though.
        </p>
        <h3 className="text-xs font-semibold mt-2">
          The Forward Process: Embracing Chaos
        </h3>
        <p>Remember how we said diffusion models add noise? Here's how:</p>
        <p className="bg-gray-800 p-2 rounded">
          q(xₜ|x₀) = N(xₜ; √(αₜ)x₀, (1-αₜ)I)
        </p>
        <p>
          Don't panic! This just means "Take an image, sprinkle some noise,
          repeat until unrecognizable." It's like playing telephone, but
          everyone's really, really bad at it.
        </p>
        <h3 className="text-xs font-semibold mt-2">
          The Reverse Process: Digital Archaelogy
        </h3>
        <p>
          Now for the magic trick - putting Humpty Dumpty back together again:
        </p>
        <p className="bg-gray-800 p-2 rounded">
          p(x₀|x₁) ≈ N(x₀; μ(x₁, 1), σ²(1)I)
        </p>
        <p>
          This is where our AI plays detective, looking at a noisy mess and
          going, "Yep, I'm pretty sure there was a cat here." It's like
          reconstructing a crime scene, if the crime was against image quality.
        </p>
        <p>
          If your brain feels like it's melting right now, you're on the right
          track! Just remember: we're teaching a computer to play "Guess That
          Image" with increasing levels of static. What could possibly go wrong?
        </p>
      </section>

      <section id="lets-build" className="space-y-4">
        <h2 className="text-sm font-semibold flex items-center">
          <Zap className="mr-2" />
          Let's Build This Thing! (What Could Go Wrong?)
        </h2>
        <p>
          Alright, intrepid AI adventurers, it's time to get our hands dirty!
          We're going to implement a diffusion model. Don't worry, I'll be right
          here holding your hand. And maybe a fire extinguisher, just in case.
        </p>
        <h3 className="text-xs font-semibold mt-2">
          Step 1: The Noise Schedule (AKA "How to Ruin a Perfectly Good Image")
        </h3>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`def cosine_beta_schedule(timesteps, s=0.008):
    """
    Create a schedule that slowly adds noise. It's like a recipe for chaos.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

betas = cosine_beta_schedule(1000)  # 1000 steps of increasingly bad decisions`}
          </code>
        </pre>
        <p>
          This function is basically saying, "Let's ruin this image, but let's
          do it <i>stylishly</i>." It's the AI equivalent of a controlled
          demolition.
        </p>
        <h3 className="text-xs font-semibold mt-2">
          Step 2: The U-Net (Not to Be Confused with a Fishing Net)
        </h3>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`class UNet(nn.Module):
    def __init__(self, channels, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # A bunch of convolutional layers and stuff go here
        # It's like a neural network lasagna

    def forward(self, x, time):
        t = self.time_mlp(time)
        # More layers, more problems
        return x  # Hopefully less noisy than when it went in`}
          </code>
        </pre>
        <p>
          This U-Net is the heart of our diffusion model. It looks at noisy
          images and goes, "Hmm, I think I see a pattern here." It's like a
          really complicated game of connect-the-dots, where half the dots are
          imaginary.
        </p>
        <h3 className="text-xs font-semibold mt-2">
          Step 3: Training (AKA "Please Work, Please Work, Please Work")
        </h3>
        <pre className="bg-gray-800 p-2 rounded text-[10px] overflow-x-auto">
          <code>
            {`for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()

        # Pick a random point in time to add noise
        t = torch.randint(0, num_timesteps, (batch.shape[0],), device=device)
        
        # Add noise to our poor, unsuspecting images
        x_t, noise = forward_diffusion_sample(batch, t, device)
        
        # Try to guess what noise we added (it's like reverse psychology for AI)
        predicted_noise = model(x_t, t)

        # Calculate how wrong we were
        loss = F.mse_loss(noise, predicted_noise)

        # Try to be less wrong next time
        loss.backward()
        optimizer.step()

        # Pray to the AI gods`}
          </code>
        </pre>
        <p>
          This is where the magic happens. We're basically playing a game of
          "Guess the Noise" with our AI, over and over again, until it gets good
          at it. It's like teaching a toddler to clean their room by repeatedly
          messing it up. Parenting 101, am I right?
        </p>
      </section>

      <section id="mind-blown" className="space-y-4">
        <h2 className="text-sm font-semibold flex items-center">
          <Sparkles className="mr-2" />
          Mind = Blown (The "Now What?" Moment)
        </h2>
        <p>
          Congratulations! If you've made it this far, you've successfully
          navigated the mind-bending world of diffusion models. Your brain
          probably feels like it's been through a washing machine, tumble dried,
          and then asked to solve a Rubik's cube. Welcome to the club!
        </p>
        <p>So, what have we learned? Well, we've discovered that:</p>
        <ul className="list-disc list-inside space-y-2">
          <li>
            Adding noise to things can actually be useful (don't try this with
            your coffee)
          </li>
          <li>
            AI can learn by un-destroying things (like a digital Sherlock
            Holmes)
          </li>
          <li>
            Math is weird, but also kind of cool (don't tell your high school
            teacher I said that)
          </li>
        </ul>
        <p>
          But here's the real kicker: diffusion models are just getting started.
          They're already creating mind-blowing images, and who knows what's
          next? Video generation? 3D model creation? A machine that can finally
          explain why my code works on my machine but not in production? (Okay,
          maybe that last one is a stretch.)
        </p>
        <p>
          As we wrap up this wild ride, remember: the next time you see an
          AI-generated image that makes you question reality, you can nod sagely
          and say, "Ah yes, diffusion models at work." And then maybe lie down
          for a bit, because let's face it, this stuff is exhausting.
        </p>
        <p>
          Until next time, keep your neurons firing and your gradients
          descending!
        </p>
      </section>

      {showScrollTop && (
        <button
          onClick={scrollToTop}
          className="fixed bottom-4 right-4 bg-gray-800 p-2 rounded-full hover:bg-gray-700 transition-colors"
          aria-label="Scroll to top"
        >
          <ChevronUp size={20} />
        </button>
      )}
    </article>
  );
}
