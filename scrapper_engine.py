'''
import requests
from bs4 import BeautifulSoup
import time
import concurrent.futures
from typing import List, Dict, Any, Union
import urllib.parse
from utils.logger import logger

class ScrappingEngine:
    """
    Robust URL scraper that guarantees consistent dictionary output.
    Never returns strings - always returns properly structured dictionaries.
    """
    
    def __init__(self, max_workers: int = 3, timeout: int = 15):
        self.max_workers = max_workers
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
    
    def scrape_single_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape single URL with guaranteed dictionary return.
        NEVER returns strings - always returns proper dict structure.
        """
        try:
            # Validate input is string
            if not isinstance(url, str):
                return self._create_error_result(str(url), "URL must be string")
            
            logger.info(f"Scraping URL: {url}")
            
            # Validate URL format
            if not self._is_valid_url(url):
                return self._create_error_result(url, "Invalid URL format")
            
            # Make HTTP request
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract article data
            article_data = self._extract_article_data(soup, url)
            
            domain = self._extract_domain(url)
            logger.info(f"{domain} domain scrapped")
            
            return article_data
            
        except Exception as e:
            error_msg = f"Scraping failed: {str(e)}"
            return self._create_error_result(url, error_msg)
    
    def _extract_article_data(self, soup: BeautifulSoup, url: str) -> Dict[str, Any]:
        """Extract structured article data - always returns dict"""
        title = self._extract_title(soup)
        content = self._extract_content(soup)
        metadata = self._extract_metadata(soup)
        
        return {
            'status': 'success',
            'url': url,
            'domain': self._extract_domain(url),
            'title': title,
            'content': content,
            'summary': self._generate_summary(content),
            'metadata': metadata,
            'content_length': len(content),
            'timestamp': time.time()
        }
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title with fallbacks"""
        # Try meta tags first
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title['content'].strip()
        
        # Try page title
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        
        # Try h1 tags
        h1 = soup.find('h1')
        if h1 and h1.get_text().strip():
            return h1.get_text().strip()
        
        return "No title found"
    
    def _extract_content(self, soup: BeautifulSoup) -> str:
        """Extract main content with fallbacks"""
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.story-content',
            'main'
        ]
        
        for selector in content_selectors:
            try:
                container = soup.select_one(selector)
                if container:
                    paragraphs = container.find_all('p')
                    if paragraphs:
                        content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                        if len(content) > 50:
                            return content
            except:
                continue
        
        # Final fallback: all paragraphs
        try:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs[:10] if p.get_text().strip()])
            return content[:3000] or "No content extracted"
        except:
            return "No content extracted"
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata - always returns dict"""
        metadata = {}
        
        # Description
        desc_meta = soup.find('meta', attrs={'name': 'description'}) or \
                   soup.find('meta', attrs={'property': 'og:description'})
        if desc_meta and desc_meta.get('content'):
            metadata['description'] = desc_meta['content'].strip()
        
        # Author
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and author_meta.get('content'):
            metadata['author'] = author_meta['content'].strip()
        
        return metadata
    
    def _generate_summary(self, content: str) -> str:
        """Generate summary - handles all input types"""
        if not content or not isinstance(content, str):
            return "No summary available"
        
        sentences = content.split('.')
        summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else content[:150]
        return summary.strip()
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL - handles all edge cases"""
        try:
            if not isinstance(url, str):
                return "unknown-domain"
            
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain or "unknown-domain"
        except:
            return "unknown-domain"
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL - robust type checking"""
        try:
            if not isinstance(url, str):
                return False
            
            parsed = urllib.parse.urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False
    
    def _create_error_result(self, url: str, error_message: str) -> Dict[str, Any]:
        """Create error result - ALWAYS returns dict, never string"""
        logger.warning(f"Scraping failed for {url}: {error_message}")
        
        return {
            'status': 'error',
            'url': str(url),  # Ensure URL is string
            'domain': self._extract_domain(str(url)),
            'error': str(error_message),  # Ensure error is string
            'timestamp': time.time()
        }
    
    def scrape_parallel(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Scrape URLs in parallel with GUARANTEED dictionary output.
        NEVER returns strings in the list.
        """
        if not urls:
            logger.warning("No URLs provided for scraping")
            return []
        
        logger.info(f"Multiple scraper initiated for {len(urls)} URLs")
        
        results = []  # This will ONLY contain dictionaries
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks
                future_to_url = {
                    executor.submit(self._safe_scrape_wrapper, url): url 
                    for url in urls[:5]  # Limit to 5 URLs for stability
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        # DOUBLE CHECK: Ensure result is dict before appending
                        if isinstance(result, dict):
                            results.append(result)
                        else:
                            # Force conversion to error dict if not dict
                            forced_result = self._create_error_result(
                                url, f"Invalid result type: {type(result)}"
                            )
                            results.append(forced_result)
                            
                    except Exception as e:
                        # Catch ANY exception and return error dict
                        error_result = self._create_error_result(url, f"Future error: {str(e)}")
                        results.append(error_result)
                        
        except Exception as e:
            logger.error(f"Parallel scraping system error: {str(e)}")
            # Create error results for all URLs
            for url in urls[:5]:
                results.append(self._create_error_result(url, f"System error: {str(e)}"))
        
        logger.info(f"Parallel scrape completed: {len(results)} results")
        return results
    
    def _safe_scrape_wrapper(self, url: str) -> Dict[str, Any]:
        """
        Safe wrapper around scrape_single_url that GUARANTEES dict return.
        This is the critical fix that prevents any string returns.
        """
        try:
            result = self.scrape_single_url(url)
            
            # CRITICAL: Ensure the result is always a dictionary
            if not isinstance(result, dict):
                return self._create_error_result(url, f"Scraper returned non-dict: {type(result)}")
            
            return result
            
        except Exception as e:
            # Catch ANY exception and return error dictionary
            return self._create_error_result(url, f"Wrapper error: {str(e)}")
    
    def get_successful_articles(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Helper method to filter successful articles"""
        return [r for r in results if isinstance(r, dict) and r.get('status') == 'success']

# Global instance with error handling
try:
    scraper_engine = ScrappingEngine()
except Exception as e:
    logger.error(f"Failed to initialize ScrappingEngine: {e}")
    # Create a fallback instance
    scraper_engine = None

def get_scraper_engine() -> ScrappingEngine:
    """Safe getter for scraper engine"""
    global scraper_engine
    if scraper_engine is None:
        scraper_engine = ScrappingEngine()
    return scraper_engine
'''
import requests
from bs4 import BeautifulSoup
import time
import concurrent.futures
from typing import List, Dict, Any, Union
import urllib.parse
from utils.logger import logger

class ScrappingEngine:
    """
    Robust URL scraper that guarantees consistent dictionary output.
    Never returns strings - always returns properly structured dictionaries.
    """
    
    def __init__(self, max_workers: int = 3, timeout: int = 15):
        self.max_workers = max_workers
        self.timeout = timeout
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    def scrape_single_url(self, url: str, claim_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Scrape single URL with guaranteed dictionary return.
        NEVER returns strings - always returns proper dict structure.
        """
        try:
            # Validate input is string
            if not isinstance(url, str):
                return self._create_error_result(str(url), "URL must be string")
            
            logger.info(f"Scraping URL: {url}")
            
            # Validate URL format
            if not self._is_valid_url(url):
                return self._create_error_result(url, "Invalid URL format")
            
            # Make HTTP request
            response = requests.get(
                url, 
                headers=self.headers, 
                timeout=self.timeout,
                allow_redirects=True,
                verify=True
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Extract article data
            article_data = self._extract_article_data(soup, url, claim_keywords)
            
            domain = self._extract_domain(url)
            logger.info(f"{domain} domain scrapped")
            
            return article_data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP error: {str(e)}"
            return self._create_error_result(url, error_msg)
        except Exception as e:
            error_msg = f"Scraping failed: {str(e)}"
            return self._create_error_result(url, error_msg)
    
    def _extract_article_data(self, soup: BeautifulSoup, url: str, claim_keywords: List[str] = None) -> Dict[str, Any]:
        """Extract structured article data - always returns dict"""
        title = self._extract_title(soup)
        content = self._extract_relevant_content(soup, claim_keywords)
        metadata = self._extract_metadata(soup)
        
        return {
            'status': 'success',
            'url': url,
            'domain': self._extract_domain(url),
            'title': title,
            'content': content,
            'summary': self._generate_summary(content),
            'metadata': metadata,
            'content_length': len(content),
            'timestamp': time.time()
        }
    
    def _extract_relevant_content(self, soup: BeautifulSoup, claim_keywords: List[str]) -> str:
        """Extract max 450 characters focusing on claim-relevant content"""
        if not claim_keywords:
            return self._extract_content_fallback(soup)
        
        # Normalize keywords
        keywords = [kw.lower() for kw in claim_keywords]
        
        paragraphs = soup.find_all('p')
        relevant_sentences = []
        total_chars = 0
        max_chars = 450
        
        # First pass: collect sentences with keywords
        for p in paragraphs:
            text = p.get_text().strip()
            if len(text) < 20:
                continue
                
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            for sentence in sentences:
                if total_chars >= max_chars:
                    break
                    
                sentence_lower = sentence.lower()
                if any(keyword in sentence_lower for keyword in keywords):
                    if total_chars + len(sentence) <= max_chars:
                        relevant_sentences.append(sentence)
                        total_chars += len(sentence)
        
        # If we have relevant content, return it
        if relevant_sentences:
            content = '. '.join(relevant_sentences)
            if len(content) > max_chars:
                content = content[:max_chars]
            return content + '.' if not content.endswith('.') else content
        
        # Fallback to regular extraction if no keywords found
        return self._extract_content_fallback(soup)
    
    def _extract_content_fallback(self, soup: BeautifulSoup) -> str:
        """Fallback content extraction when no keywords match"""
        content_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.story-content',
            'main'
        ]
        
        for selector in content_selectors:
            try:
                container = soup.select_one(selector)
                if container:
                    paragraphs = container.find_all('p')
                    if paragraphs:
                        content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                        if len(content) > 50:
                            return content[:450]
            except:
                continue
        
        # Final fallback
        try:
            paragraphs = soup.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs[:10] if p.get_text().strip()])
            return content[:450] or "No content extracted"
        except:
            return "No content extracted"
    
    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract article title with fallbacks"""
        # Try meta tags first
        meta_title = soup.find('meta', property='og:title')
        if meta_title and meta_title.get('content'):
            return meta_title['content'].strip()
        
        # Try page title
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        
        # Try h1 tags
        h1 = soup.find('h1')
        if h1 and h1.get_text().strip():
            return h1.get_text().strip()
        
        return "No title found"
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """Extract metadata - always returns dict"""
        metadata = {}
        
        # Description
        desc_meta = soup.find('meta', attrs={'name': 'description'}) or \
                   soup.find('meta', attrs={'property': 'og:description'})
        if desc_meta and desc_meta.get('content'):
            metadata['description'] = desc_meta['content'].strip()
        
        # Author
        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta and author_meta.get('content'):
            metadata['author'] = author_meta['content'].strip()
        
        return metadata
    
    def _generate_summary(self, content: str) -> str:
        """Generate summary - handles all input types"""
        if not content or not isinstance(content, str):
            return "No summary available"
        
        sentences = content.split('.')
        summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else content[:150]
        return summary.strip()
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL - handles all edge cases"""
        try:
            if not isinstance(url, str):
                return "unknown-domain"
            
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc.lower()
            
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain or "unknown-domain"
        except:
            return "unknown-domain"
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL - robust type checking"""
        try:
            if not isinstance(url, str):
                return False
            
            parsed = urllib.parse.urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except:
            return False
    
    def _create_error_result(self, url: str, error_message: str) -> Dict[str, Any]:
        """Create error result - ALWAYS returns dict, never string"""
        logger.warning(f"Scraping failed for {url}: {error_message}")
        
        return {
            'status': 'error',
            'url': str(url),  # Ensure URL is string
            'domain': self._extract_domain(str(url)),
            'error': str(error_message),  # Ensure error is string
            'timestamp': time.time()
        }
    
    def scrape_parallel(self, urls: List[str], claim_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        Scrape URLs in parallel with GUARANTEED dictionary output.
        NEVER returns strings in the list.
        """
        if not urls:
            logger.warning("No URLs provided for scraping")
            return []
        
        logger.info(f"Multiple scraper initiated for {len(urls)} URLs")
        
        results = []  # This will ONLY contain dictionaries
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks
                future_to_url = {
                    executor.submit(self._safe_scrape_wrapper, url, claim_keywords): url 
                    for url in urls[:5]  # Limit to 5 URLs for stability
                }
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_url):
                    url = future_to_url[future]
                    try:
                        result = future.result()
                        # DOUBLE CHECK: Ensure result is dict before appending
                        if isinstance(result, dict):
                            results.append(result)
                        else:
                            # Force conversion to error dict if not dict
                            forced_result = self._create_error_result(
                                url, f"Invalid result type: {type(result)}"
                            )
                            results.append(forced_result)
                            
                    except Exception as e:
                        # Catch ANY exception and return error dict
                        error_result = self._create_error_result(url, f"Future error: {str(e)}")
                        results.append(error_result)
                        
        except Exception as e:
            logger.error(f"Parallel scraping system error: {str(e)}")
            # Create error results for all URLs
            for url in urls[:5]:
                results.append(self._create_error_result(url, f"System error: {str(e)}"))
        
        logger.info(f"Parallel scrape completed: {len(results)} results")
        return results
    
    def _safe_scrape_wrapper(self, url: str, claim_keywords: List[str] = None) -> Dict[str, Any]:
        """
        Safe wrapper around scrape_single_url that GUARANTEES dict return.
        This is the critical fix that prevents any string returns.
        """
        try:
            result = self.scrape_single_url(url, claim_keywords)
            
            # CRITICAL: Ensure the result is always a dictionary
            if not isinstance(result, dict):
                return self._create_error_result(url, f"Scraper returned non-dict: {type(result)}")
            
            return result
            
        except Exception as e:
            # Catch ANY exception and return error dictionary
            return self._create_error_result(url, f"Wrapper error: {str(e)}")
    
    def get_successful_articles(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Helper method to filter successful articles"""
        return [r for r in results if isinstance(r, dict) and r.get('status') == 'success']

# Global instance with error handling
try:
    scraper_engine = ScrappingEngine()
except Exception as e:
    logger.error(f"Failed to initialize ScrappingEngine: {e}")
    # Create a fallback instance
    scraper_engine = None

def get_scraper_engine() -> ScrappingEngine:
    """Safe getter for scraper engine"""
    global scraper_engine
    if scraper_engine is None:
        scraper_engine = ScrappingEngine()
    return scraper_engine