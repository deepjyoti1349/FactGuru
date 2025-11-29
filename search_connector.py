'''

from ddgs import DDGS
import time
from utils.logger import logger


class SearchConnector:
    def __init__(self, max_results=7):
        self.max_results = max_results
    
    def search(self, claim):
        """Execute a single search and return formatted results"""
        try:
            results = []
            # Fixed: use the claim variable instead of string "claim"
            search_results = DDGS().text(claim, max_results=self.max_results)
            
            for result in search_results:
                results.append({
                    "title": result.get("title", "no title"),
                    "url": result.get("href", "no URL"),
                    "snippet": result.get("body", "no snippet"),
                    "source": "duckduckgo",
                })
            
            # Fixed: moved outside the for loop
            if results:
                logger.info(f"Found {len(results)} results")
                return results
            else:
                logger.warning(f"No results found for: {claim}")
                return []
                
        except Exception as e:
            logger.warning(f"Search error: {e}")
            return []  # Fixed: always return a list
    
    def search_driver(self, claim):
        """Main search method with retry logic"""
        logger.info(f"Searching for: {claim}")
        
        for attempt in range(3):
            try:
                results = self.search(claim)
                
                # If we got results (even empty), return them
                if results is not None:  # search() now always returns a list
                    return results
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(1)  # Fixed: added delay between retries
                    continue
        
        logger.warning("All search attempts failed")
        return []
    
    '''
from ddgs import DDGS
import time
from utils.logger import logger


class SearchConnector:
    def __init__(self, max_results=7):
        self.max_results = max_results
    
    def search_ddg_only(self, claim):
        """Execute search using only DuckDuckGo"""
        try:
            results = []
            logger.info(f"Searching DuckDuckGo for: {claim}")
            
            # Use DuckDuckGo directly with error handling
            with DDGS() as ddgs:
                search_results = ddgs.text(claim, max_results=self.max_results)
                
                for result in search_results:
                    results.append({
                        "title": result.get("title", "no title"),
                        "url": result.get("href", "no URL"),
                        "snippet": result.get("body", "no snippet"),
                        "source": "duckduckgo",
                    })
            
            if results:
                logger.info(f"Found {len(results)} results from DuckDuckGo")
                return results
            else:
                logger.warning(f"No results found for: {claim}")
                return []
                
        except Exception as e:
            logger.warning(f"DuckDuckGo search error: {e}")
            return []
    
    def search_driver(self, claim):
        """Main search method with retry logic - DuckDuckGo only"""
        logger.info(f"Searching for: {claim}")
        
        for attempt in range(3):
            try:
                results = self.search_ddg_only(claim)
                
                if results:  # If we got results, return them
                    return results
                else:
                    logger.info(f"Attempt {attempt + 1}: No results, retrying...")
                    
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < 2:  # Don't sleep on last attempt
                time.sleep(2)  # Longer delay between retries
        
        logger.warning("All DuckDuckGo search attempts completed")
        return []

    # Optional: If you want to completely replace the old search method
    def search(self, claim):
        """Alias for DuckDuckGo only search"""
        return self.search_ddg_only(claim)