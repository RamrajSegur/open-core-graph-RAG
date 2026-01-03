"""Parser for extracting content from webpages (HTML)."""

import logging
from typing import Optional
from urllib.parse import urlparse

from .base_parser import BaseParser, DocumentType, ParsedDocument

logger = logging.getLogger(__name__)


class WebpageParser(BaseParser):
    """Parser for extracting text content from HTML webpages."""

    def __init__(self, timeout: int = 10):
        """
        Initialize the webpage parser.

        Args:
            timeout: Timeout in seconds for fetching webpages
        """
        super().__init__()
        self.timeout = timeout
        try:
            import requests
            from bs4 import BeautifulSoup
            self.requests = requests
            self.BeautifulSoup = BeautifulSoup
            self.available = True
        except ImportError:
            logger.warning(
                "requests or beautifulsoup4 not installed. "
                "Webpage parsing will be limited."
            )
            self.available = False

    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given input.

        Args:
            file_path: URL or path to HTML file

        Returns:
            True if this parser can handle the input
        """
        # Check if it's a URL
        if file_path.startswith(('http://', 'https://')):
            return True
        
        # Check if it's an HTML file
        if file_path.lower().endswith('.html'):
            return True
        
        return False

    def parse(self, file_path: str) -> ParsedDocument:
        """
        Parse a webpage from URL or HTML file.

        Args:
            file_path: URL (http/https) or path to HTML file

        Returns:
            ParsedDocument with extracted content

        Raises:
            ValueError: If webpage cannot be fetched or parsed
            FileNotFoundError: If local HTML file doesn't exist
        """
        if not self.available:
            raise ImportError(
                "requests and beautifulsoup4 are required for webpage parsing. "
                "Install with: pip install requests beautifulsoup4"
            )

        # Handle URL
        if file_path.startswith(('http://', 'https://')):
            return self._parse_url(file_path)
        
        # Handle local HTML file
        if file_path.lower().endswith('.html'):
            return self._parse_html_file(file_path)
        
        raise ValueError(f"Unsupported format: {file_path}")

    def _parse_url(self, url: str) -> ParsedDocument:
        """
        Fetch and parse a webpage from URL.

        Args:
            url: Webpage URL

        Returns:
            ParsedDocument with extracted content

        Raises:
            ValueError: If webpage cannot be fetched
        """
        try:
            logger.info(f"Fetching webpage: {url}")
            
            # Fetch the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                             'AppleWebKit/537.36'
            }
            response = self.requests.get(
                url,
                timeout=self.timeout,
                headers=headers
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = self.BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            title = soup.title.string if soup.title else "Unknown"
            description = ""
            meta_desc = soup.find('meta', {'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                description = meta_desc.get('content')
            
            # Extract main content
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
            
            # Create metadata
            metadata = {
                'url': url,
                'title': title,
                'description': description,
                'content_type': response.headers.get('content-type', 'text/html'),
                'charset': response.encoding or 'utf-8',
                'status_code': response.status_code,
            }
            
            logger.info(f"Successfully extracted {len(text)} characters from {url}")
            
            return ParsedDocument(
                file_path=url,
                file_type=DocumentType.HTML,
                raw_text=text,
                metadata=metadata,
                pages=[text]  # Webpages are treated as single page
            )
        
        except self.requests.RequestException as e:
            logger.error(f"Failed to fetch webpage {url}: {e}")
            raise ValueError(f"Cannot fetch webpage: {e}") from e
        except Exception as e:
            logger.error(f"Error parsing webpage {url}: {e}")
            raise ValueError(f"Cannot parse webpage: {e}") from e

    def _parse_html_file(self, file_path: str) -> ParsedDocument:
        """
        Parse a local HTML file.

        Args:
            file_path: Path to HTML file

        Returns:
            ParsedDocument with extracted content

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file cannot be parsed
        """
        try:
            path = self._validate_file_exists(file_path)
            logger.info(f"Parsing HTML file: {file_path}")
            
            # Read HTML file
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            # Parse HTML
            soup = self.BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata
            title = soup.title.string if soup.title else path.stem
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace
            text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
            
            # Create metadata
            metadata = {
                'filename': path.name,
                'file_path': str(path),
                'title': title,
                'file_size': path.stat().st_size,
            }
            
            logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
            
            return ParsedDocument(
                file_path=str(path),
                file_type=DocumentType.HTML,
                raw_text=text,
                metadata=metadata,
                pages=[text]  # Local HTML files are treated as single page
            )
        
        except FileNotFoundError as e:
            logger.error(f"HTML file not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error parsing HTML file {file_path}: {e}")
            raise ValueError(f"Cannot parse HTML file: {e}") from e
