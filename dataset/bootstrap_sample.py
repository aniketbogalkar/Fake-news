"""
Bootstrap Sample Dataset Generator for NewsCheck AI
Generates authentic-format True.csv and Fake.csv with diverse real and fake news samples.
Allows immediate local model training and verification testing.
"""

import os
import pandas as pd

DATASET_DIR = os.path.dirname(__file__)

REAL_ARTICLES = [
    {
        "title": "Indian Space Research Organisation successfully tests semi-cryogenic engine for next-gen launch vehicles",
        "text": "The Indian Space Research Organisation (ISRO) on Thursday announced the successful hot test of an intermediate configuration of the semi-cryogenic engine at its Propulsion Complex in Mahendragiri, Tamil Nadu. The milestone advances India's heavy-lift launch capabilities. The space agency stated that the test lasted 150 seconds and all parameters met mission requirements.",
        "subject": "technology",
        "date": "March 15, 2024"
    },
    {
        "title": "Reserve Bank of India keeps repo rate unchanged at 6.5 percent amid inflation vigilance",
        "text": "The Monetary Policy Committee of the Reserve Bank of India, headed by Governor Shaktikanta Das, has decided by a 5 to 1 majority to keep the policy repo rate unchanged at 6.50 per cent. The central bank maintained its monetary stance on withdrawal of accommodation to ensure that inflation progressively aligns with the target while supporting domestic economic growth.",
        "subject": "politics",
        "date": "April 5, 2024"
    },
    {
        "title": "Reuters: European Union antitrust regulators investigate tech giants under Digital Markets Act",
        "text": "BRUSSELS (Reuters) - European Union antitrust regulators have opened investigations into whether Apple, Alphabet's Google, and Meta Platforms have breached landmark digital regulations aimed at curbing Big Tech's market power. The European Commission said it suspects that measures put in place by these gatekeepers fall short of effective compliance with the Digital Markets Act.",
        "subject": "worldnews",
        "date": "March 25, 2024"
    },
    {
        "title": "BBC News: Global renewable energy capacity expanded by fifty percent in record surge",
        "text": "The International Energy Agency reported that global renewable capacity additions increased by almost 50% to nearly 510 gigawatts in 2023, marking the fastest growth rate in two decades. Solar PV accounted for three-quarters of the worldwide additions. The IEA noted that while expansion is strong in China and Europe, financing remains a challenge for emerging economies.",
        "subject": "worldnews",
        "date": "January 11, 2024"
    },
    {
        "title": "The Hindu: Supreme Court of India strikes down electoral bonds scheme as unconstitutional",
        "text": "In a unanimous landmark verdict, a five-judge Constitution Bench of the Supreme Court of India led by Chief Justice of India D.Y. Chandrachud struck down the Union government's 2018 Electoral Bonds scheme. The court held that the anonymous funding scheme violates the fundamental Right to Information under Article 19(1)(a) of the Constitution.",
        "subject": "politics",
        "date": "February 15, 2024"
    },
    {
        "title": "World Health Organization warns against rising global measles cases due to missed vaccinations",
        "text": "GENEVA - The World Health Organization and UNICEF have issued an urgent warning over a surge in measles infections worldwide, driven by declining childhood vaccination coverage during recent humanitarian crises and the pandemic. Health officials urged governments to conduct catch-up vaccination drives immediately to prevent large-scale outbreaks.",
        "subject": "worldnews",
        "date": "February 20, 2024"
    },
    {
        "title": "Times of India: New Delhi Metro expands automated driverless train operations on Pink Line",
        "text": "NEW DELHI - The Delhi Metro Rail Corporation (DMRC) has expanded Communication Based Train Control (CBTC) signaling across the Pink Line, enabling driverless train operations on over 59 kilometers. Officials stated the system improves headway, passenger safety, and energy efficiency across the national capital region network.",
        "subject": "news",
        "date": "May 10, 2024"
    },
    {
        "title": "Reuters: United Nations Security Council adopts resolution demanding immediate Gaza ceasefire",
        "text": "UNITED NATIONS (Reuters) - The United Nations Security Council passed a resolution on Monday demanding an immediate ceasefire for the Muslim holy month of Ramadan, the immediate and unconditional release of all hostages, and the urgent need to expand the flow of humanitarian assistance into Gaza.",
        "subject": "worldnews",
        "date": "March 25, 2024"
    },
    {
        "title": "James Webb Space Telescope detects water vapor in terrestrial planet-forming zone",
        "text": "Astronomers using NASA's James Webb Space Telescope have detected water vapor in the inner disk of the planetary system PDS 70, where rocky terrestrial planets are actively forming. The discovery provides evidence that habitable planets could acquire water during their earliest stages of formation.",
        "subject": "technology",
        "date": "July 24, 2023"
    },
    {
        "title": "Ministry of Finance reports gross GST revenue collections crossed 1.8 lakh crore in milestone",
        "text": "India's gross Goods and Services Tax (GST) collections reached a record high of Rs 1.87 lakh crore in April, driven by strong economic momentum, domestic consumption, and improved compliance measures, according to data released by the Ministry of Finance.",
        "subject": "politics",
        "date": "May 1, 2024"
    }
]

FAKE_ARTICLES = [
    {
        "title": "NASA confirms 15 days of total darkness across Earth due to planetary alignment",
        "text": "NASA has confirmed that the Earth will experience 15 consecutive days of complete darkness starting next week. The space agency stated that a rare cosmic alignment between Venus and Jupiter will cause massive solar flares that will black out the Sun completely. Citizens are advised to stockpile candles, canned food, and stay indoors until daylight returns.",
        "subject": "News",
        "date": "April 1, 2024"
    },
    {
        "title": "UNESCO declares national anthem of India as the best national anthem in the world",
        "text": "Breaking news viral on social media: UNESCO has officially voted and selected Jana Gana Mana as the best national anthem in the entire world. In a top-secret vote conducted in Paris yesterday, representatives from 195 countries unanimously awarded India the first prize, defeating USA and UK.",
        "subject": "politics",
        "date": "August 14, 2023"
    },
    {
        "title": "New 2000 rupee currency notes contain secret radioactive Nano GPS tracking chips",
        "text": "A shocking revelation has emerged regarding new currency banknotes. Every note is embedded with a secret NGC (Nano GPS Chip) that communicates with Indian spy satellites. The signal can be tracked even if the currency notes are buried 120 meters underground in tax havens. Corrupt hoarders have no escape now as signals ping every 10 minutes.",
        "subject": "News",
        "date": "November 10, 2023"
    },
    {
        "title": "World Health Organization announces mandatory global lockdown starting next month under secret treaty",
        "text": "Leaked documents reveal that the WHO has ordered all member nations to impose a 90-day martial law lockdown starting next month. Citizens will not be permitted to leave their residences, bank accounts will be seized, and all flights will be suspended under a new globalist pandemic accord ratified behind closed doors.",
        "subject": "worldnews",
        "date": "March 12, 2024"
    },
    {
        "title": "Drinking boiled garlic water completely cures cancer and eliminates all viral infections within 24 hours",
        "text": "Big Pharma does not want you to know this miraculous secret remedy: boiling 8 cloves of garlic with lemon juice and drinking it on an empty stomach will cure all stages of malignant tumors in just 24 hours. Top unnamed doctors confirm hospitals are hiding this simple household spice cure to protect billion-dollar chemotherapy profits.",
        "subject": "health",
        "date": "January 5, 2024"
    },
    {
        "title": "Secret underground pyramids discovered under Antarctic ice shelf by anonymous expedition",
        "text": "Satellite radar imagery leaked from military servers shows gigantic ancient pyramid structures hidden underneath 2 miles of Antarctic ice. Researchers claim ancient extraterrestrial civilizations constructed power generators there over 100,000 years ago before an abrupt polar shift froze the continent.",
        "subject": "conspiracy",
        "date": "May 2, 2023"
    },
    {
        "title": "Scientists admit 5G mobile towers generate electromagnetic waves that control human thoughts and weather",
        "text": "Whistleblower engineers have confessed on an underground podcast that high-frequency 5G cell towers are actually classified cloud-seeding and neural broadcast transmitters. The government is allegedly manipulating weather patterns and mood regulation of urban populations through wireless telecom frequency bands.",
        "subject": "conspiracy",
        "date": "June 18, 2023"
    },
    {
        "title": "Supreme Court abolishes all income taxes and orders government to refund ten years of deductions",
        "text": "In a sudden dramatic emergency midnight hearing, the Supreme Court ruled that personal income tax is completely illegal under common law. All employers have been directed to stop withholding taxes immediately and tax departments must refund all taxes collected since 2014 directly into citizen bank accounts with 12 percent interest.",
        "subject": "politics",
        "date": "July 4, 2023"
    },
    {
        "title": "WhatsApp forwards to be monitored and charged three rupees per forwarded message by telecom companies",
        "text": "Forwarded as received: Urgent notice from ministry of communications. Starting tomorrow midnight, all forward messages on messaging apps will be charged 3 rupees each directly from SIM card balances. Only messages with blue tick confirmations will be free. Forward this message to 10 groups immediately to activate lifetime free service.",
        "subject": "News",
        "date": "September 15, 2023"
    },
    {
        "title": "Astronomers discover giant alien mothership orbiting Jupiter cloaked as moon",
        "text": "Leaked telescope data from an amateur astronomer collective proves that one of Jupiter's outer moons is actually a massive artificial spacecraft measuring 500 kilometers across. Officials at major space observatories have reportedly signed non-disclosure agreements under national defense directives.",
        "subject": "conspiracy",
        "date": "August 22, 2023"
    }
]


def generate_bootstrap_dataset(force=False):
    os.makedirs(DATASET_DIR, exist_ok=True)
    true_path = os.path.join(DATASET_DIR, "True.csv")
    fake_path = os.path.join(DATASET_DIR, "Fake.csv")

    if not force and os.path.exists(true_path) and os.path.exists(fake_path):
        print(f"[Dataset] Existing datasets found at {true_path} and {fake_path}.")
        return

    # Expand the samples with minor variations to make adequate training data if needed
    expanded_real = []
    expanded_fake = []
    
    # Replicate and perturb to create ~100 balanced samples for bootstrapping
    for i in range(10):
        for item in REAL_ARTICLES:
            expanded_real.append({
                "title": item["title"] if i == 0 else f"{item['title']} - Report Update {i}",
                "text": item["text"],
                "subject": item["subject"],
                "date": item["date"]
            })
        for item in FAKE_ARTICLES:
            expanded_fake.append({
                "title": item["title"] if i == 0 else f"{item['title']} - Shocking Alert {i}",
                "text": item["text"],
                "subject": item["subject"],
                "date": item["date"]
            })

    df_real = pd.DataFrame(expanded_real)
    df_fake = pd.DataFrame(expanded_fake)

    df_real.to_csv(true_path, index=False)
    df_fake.to_csv(fake_path, index=False)
    print(f"[Dataset] Successfully generated sample dataset: {len(df_real)} real and {len(df_fake)} fake articles.")
    print(f"[Dataset] True.csv saved to {true_path}")
    print(f"[Dataset] Fake.csv saved to {fake_path}")


if __name__ == "__main__":
    generate_bootstrap_dataset(force=True)
