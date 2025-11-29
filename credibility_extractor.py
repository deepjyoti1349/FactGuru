class CredibilityExtractor:
    def __init__(self):
        self.known_sites = {
            'abcnews.go.com': 0.78,
            'aljazeera.com': 0.82,
            'apnews.com': 0.93,
            'bbc.com': 0.90,
            'bloomberg.com': 0.85,
            'cbsnews.com': 0.77,
            'cnbc.com': 0.83,
            'cnn.com': 0.75,
            'dw.com': 0.86,
            'economist.com': 0.88,
            'forbes.com': 0.80,
            'foxnews.com': 0.65,
            'ft.com': 0.89,
            'guardian.co.uk': 0.85,
            'huffpost.com': 0.72,
            'latimes.com': 0.79,
            'nbcnews.com': 0.78,
            'news.sky.com': 0.76,
            'newsweek.com': 0.74,
            'npr.org': 0.88,
            'nytimes.com': 0.82,
            'reuters.com': 0.95,
            'theguardian.com': 0.85,
            'time.com': 0.81,
            'usatoday.com': 0.76,
            'washingtonpost.com': 0.78,
            'wsj.com': 0.87,
            'xinhuanet.com': 0.60,
            'aajtak.in': 0.70,
            'abplive.com': 0.68,
            'aninews.in': 0.72,
            'bdaily.in': 0.65,
            'business-standard.com': 0.82,
            'deccanchronicle.com': 0.74,
            'dnaindia.com': 0.73,
            'economicstimes.com': 0.84,
            'financialexpress.com': 0.79,
            'firstpost.com': 0.76,
            'greaterkashmir.com': 0.71,
            'hindustantimes.com': 0.83,
            'indianexpress.com': 0.85,
            'indiatoday.in': 0.80,
            'indiatvnews.com': 0.69,
            'livemint.com': 0.86,
            'ndtv.com': 0.87,
            'news18.com': 0.78,
            'news24online.com': 0.67,
            'outlookindia.com': 0.75,
            'patrika.com': 0.72,
            'punjabkesari.in': 0.66,
            'republicworld.com': 0.62,
            'scroll.in': 0.81,
            'thehindu.com': 0.92,
            'thelogicalindian.com': 0.73,
            'thequint.com': 0.77,
            'timesofindia.com': 0.88,
            'zeenews.com': 0.71,
            'amaralok.com': 0.65,
            'asomiyapratidin.in': 0.78,
            'assamjournal.com': 0.68,
            'assamtribune.com': 0.85,
            'bartabangla.com': 0.62,
            'dainikagradoot.com': 0.75,
            'dainikasam.com': 0.73,
            'dainikjanambhumi.com': 0.77,
            'gnmnews.com': 0.66,
            'guwahatiplus.com': 0.71,
            'journalsworld.com': 0.64,
            'karimganjtimes.com': 0.69,
            'mumbainews.net': 0.63,
            'neindia.com': 0.79,
            'nenow.in': 0.82,
            'newsofassam.com': 0.74,
            'northeasttoday.in': 0.80,
            'pragnews.com': 0.83,
            'pratidintime.com': 0.76,
            'purvanchalpradesh.com': 0.67,
            'samaylive.com': 0.70,
            'sentinelassam.com': 0.84,
            'sikkimexpress.com': 0.72,
            'thehillstimes.in': 0.81,
            'thenortheasttoday.com': 0.78,
            'timesofassam.com': 0.75
        }
        self.unknown_sites = {}
    
    def get_website_score(self, website_name):
        # Check if website is in known sites
        if website_name in self.known_sites:
            # Return website and its score
            return {'website': website_name, 'score': self.known_sites[website_name]}
        else:
            # Add to unknown sites with default score
            self.unknown_sites[website_name] = 0.5
            # Return website with default score
            return {'website': website_name, 'score': 0.5}
    
    def analyze_multiple_websites(self, website_list):
        # List to store all results
        results = []
        # Loop through each website
        for website in website_list:
            # Get score for current website
            website_result = self.get_website_score(website)
            # Add to results list
            results.append(website_result)
        # Return all results
        return results
    
    def extract_domain(self, url):
        # Remove http:// and https://
        clean_url = url.replace('https://', '').replace('http://', '')
        # Remove www.
        clean_url = clean_url.replace('www.', '')
        # Split by / and take first part
        domain_parts = clean_url.split('/')
        # Return the domain name
        return domain_parts[0]