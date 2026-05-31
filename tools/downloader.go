package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"golang.org/x/net/html"
)

// Config holds all configurable options for the downloader
type Config struct {
	URL            string
	OutputDir      string
	Extensions     []string
	Keywords       []string
	Exclude        []*regexp.Regexp // skip URLs matching any of these patterns
	Delay          time.Duration
	Yes            bool // skip confirmation prompt
	Retries        int
	RequestTimeout time.Duration // overall per-attempt timeout
	StallTimeout   time.Duration // abort if no bytes received for this long
}

// DefaultExtensions is the default set of file extensions to download
var DefaultExtensions = []string{
	".pdf", ".ppt", ".pptx", ".docx", ".doc",
	".zip", ".mp4", ".webm", ".csv", ".txt", ".ipynb",
}

// Resource represents a downloadable resource
type Resource struct {
	URL  string
	Text string
	Href string
}

// Downloader handles fetching, parsing, and downloading resources from a web page
type Downloader struct {
	config          Config
	client          *http.Client
	downloadedFiles []string
	failedFiles     []string
	mu              sync.Mutex
}

// NewDownloader creates a new Downloader instance from a Config
func NewDownloader(cfg Config) *Downloader {
	// Expand home directory if needed
	if strings.HasPrefix(cfg.OutputDir, "~") {
		home, _ := os.UserHomeDir()
		cfg.OutputDir = filepath.Join(home, cfg.OutputDir[1:])
	}

	os.MkdirAll(cfg.OutputDir, 0755)

	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   15 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout:   15 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
		ExpectContinueTimeout: 5 * time.Second,
		IdleConnTimeout:       60 * time.Second,
	}

	return &Downloader{
		config: cfg,
		client: &http.Client{Transport: transport},
	}
}

// FetchPage fetches the page at the configured URL
func (d *Downloader) FetchPage() (string, error) {
	log.Printf("Fetching %s...\n", d.config.URL)

	req, err := http.NewRequest("GET", d.config.URL, nil)
	if err != nil {
		return "", err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

	resp, err := d.client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to fetch page: status code %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return string(body), nil
}

// ExtractLinks extracts all anchor links from HTML content
func (d *Downloader) ExtractLinks(htmlContent string) []Resource {
	if htmlContent == "" {
		return []Resource{}
	}

	doc, err := html.Parse(strings.NewReader(htmlContent))
	if err != nil {
		log.Printf("Error parsing HTML: %v\n", err)
		return []Resource{}
	}

	var links []Resource
	var extract func(*html.Node)

	extract = func(n *html.Node) {
		if n.Type == html.ElementNode && n.Data == "a" {
			var href string
			for _, attr := range n.Attr {
				if attr.Key == "href" {
					href = attr.Val
					break
				}
			}

			text := extractText(n)

			if href != "" && !strings.HasPrefix(href, "#") {
				fullURL := resolveURL(d.config.URL, href)
				links = append(links, Resource{URL: fullURL, Text: text, Href: href})
			}
		}
		for c := n.FirstChild; c != nil; c = c.NextSibling {
			extract(c)
		}
	}

	extract(doc)
	return links
}

// extractText extracts text content from an HTML node
func extractText(n *html.Node) string {
	var text string
	var extract func(*html.Node)
	extract = func(node *html.Node) {
		if node.Type == html.TextNode {
			text += strings.TrimSpace(node.Data)
		}
		for c := node.FirstChild; c != nil; c = c.NextSibling {
			extract(c)
		}
	}
	extract(n)
	return strings.TrimSpace(text)
}

// resolveURL resolves a relative URL against a base URL
func resolveURL(baseURL, href string) string {
	base, err := url.Parse(baseURL)
	if err != nil {
		return href
	}
	ref, err := url.Parse(href)
	if err != nil {
		return href
	}
	return base.ResolveReference(ref).String()
}

// FilterResources filters links by configured extensions and keyword patterns
func (d *Downloader) FilterResources(links []Resource) []Resource {
	extSet := make(map[string]bool, len(d.config.Extensions))
	for _, ext := range d.config.Extensions {
		extSet[strings.ToLower(ext)] = true
	}

	var resources []Resource
links_loop:
	for _, link := range links {
		urlLower := strings.ToLower(link.URL)

		for _, ex := range d.config.Exclude {
			if ex.MatchString(urlLower) {
				continue links_loop
			}
		}

		matchesExt := false
		for ext := range extSet {
			if strings.HasSuffix(urlLower, ext) {
				matchesExt = true
				break
			}
		}

		matchesKeyword := false
		for _, kw := range d.config.Keywords {
			if strings.Contains(urlLower, strings.ToLower(kw)) {
				matchesKeyword = true
				break
			}
		}

		if matchesExt || matchesKeyword {
			resources = append(resources, link)
		}
	}
	return resources
}

// progressReader wraps an io.Reader and renders a progress bar while tracking
// the last time bytes were received so a watchdog can abort on stalls.
type progressReader struct {
	reader     io.Reader
	total      int64
	downloaded int64 // atomic
	lastTick   int64 // atomic; unix nanos of last read with n>0
	filename   string
	lastRender time.Time
}

func (p *progressReader) Read(b []byte) (int, error) {
	n, err := p.reader.Read(b)
	if n > 0 {
		atomic.AddInt64(&p.downloaded, int64(n))
		atomic.StoreInt64(&p.lastTick, time.Now().UnixNano())
		if time.Since(p.lastRender) > 100*time.Millisecond {
			p.render()
			p.lastRender = time.Now()
		}
	}
	return n, err
}

func (p *progressReader) render() {
	downloaded := atomic.LoadInt64(&p.downloaded)
	name := p.filename
	if len(name) > 30 {
		name = name[:27] + "..."
	}
	const barWidth = 30
	if p.total > 0 {
		pct := float64(downloaded) / float64(p.total)
		if pct > 1 {
			pct = 1
		}
		filled := int(pct * barWidth)
		bar := strings.Repeat("=", filled) + strings.Repeat(" ", barWidth-filled)
		fmt.Fprintf(os.Stderr, "\r  [%s] %5.1f%% %7.2f/%.2f MB %s",
			bar, pct*100,
			float64(downloaded)/(1024*1024),
			float64(p.total)/(1024*1024),
			name)
	} else {
		fmt.Fprintf(os.Stderr, "\r  [downloading] %7.2f MB %s",
			float64(downloaded)/(1024*1024), name)
	}
}

func (p *progressReader) finish() {
	p.render()
	fmt.Fprintln(os.Stderr)
}

// DownloadFile downloads a single file with retries, per-attempt timeout,
// stall detection, and a progress bar.
func (d *Downloader) DownloadFile(resourceURL string) bool {
	filename := path.Base(resourceURL)
	if filename == "" || filename == "/" {
		filename = fmt.Sprintf("resource_%d", time.Now().Unix())
	}
	filename = sanitizeFilename(filename)

	destPath := filepath.Join(d.config.OutputDir, filename)

	if _, err := os.Stat(destPath); err == nil {
		log.Printf("File already exists: %s\n", filename)
		return true
	}

	retries := d.config.Retries
	if retries < 1 {
		retries = 1
	}

	var lastErr error
	for attempt := 1; attempt <= retries; attempt++ {
		if attempt > 1 {
			backoff := time.Duration(attempt) * 2 * time.Second
			log.Printf("Retry %d/%d for %s after %v (last error: %v)\n",
				attempt, retries, filename, backoff, lastErr)
			time.Sleep(backoff)
		}

		log.Printf("Downloading: %s\n", filename)
		ok, err := d.downloadOnce(resourceURL, destPath, filename)
		if ok {
			d.mu.Lock()
			d.downloadedFiles = append(d.downloadedFiles, filename)
			d.mu.Unlock()
			time.Sleep(d.config.Delay)
			return true
		}
		lastErr = err
		os.Remove(destPath)
	}

	log.Printf("Giving up on %s after %d attempts: %v\n", resourceURL, retries, lastErr)
	d.recordFailure(resourceURL)
	return false
}

func (d *Downloader) downloadOnce(resourceURL, destPath, filename string) (bool, error) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if d.config.RequestTimeout > 0 {
		var c2 context.CancelFunc
		ctx, c2 = context.WithTimeout(ctx, d.config.RequestTimeout)
		defer c2()
	}

	req, err := http.NewRequestWithContext(ctx, "GET", resourceURL, nil)
	if err != nil {
		return false, err
	}
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")

	resp, err := d.client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("status code %d", resp.StatusCode)
	}

	file, err := os.Create(destPath)
	if err != nil {
		return false, err
	}
	defer file.Close()

	pr := &progressReader{
		reader:   resp.Body,
		total:    resp.ContentLength,
		filename: filename,
		lastTick: time.Now().UnixNano(),
	}

	// Stall watchdog: cancel the request if no bytes received within StallTimeout.
	if d.config.StallTimeout > 0 {
		done := make(chan struct{})
		defer close(done)
		go func() {
			ticker := time.NewTicker(2 * time.Second)
			defer ticker.Stop()
			for {
				select {
				case <-done:
					return
				case <-ticker.C:
					last := time.Unix(0, atomic.LoadInt64(&pr.lastTick))
					if time.Since(last) > d.config.StallTimeout {
						cancel()
						return
					}
				}
			}
		}()
	}

	written, err := io.Copy(file, pr)
	pr.finish()
	if err != nil {
		return false, err
	}

	log.Printf("Downloaded: %s (%.2f MB)\n", filename, float64(written)/(1024*1024))
	return true, nil
}

func (d *Downloader) recordFailure(resourceURL string) {
	d.mu.Lock()
	d.failedFiles = append(d.failedFiles, resourceURL)
	d.mu.Unlock()
}

// sanitizeFilename removes characters that are unsafe in filenames
func sanitizeFilename(filename string) string {
	var result strings.Builder
	for _, r := range filename {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') ||
			(r >= '0' && r <= '9') || r == '.' || r == '_' || r == '-' {
			result.WriteRune(r)
		}
	}
	return strings.TrimRight(result.String(), ".")
}

// Run executes the full fetch-parse-filter-download pipeline
func (d *Downloader) Run() {
	log.Printf("Starting downloader\n")
	log.Printf("URL: %s\n", d.config.URL)
	log.Printf("Output directory: %s\n", d.config.OutputDir)

	pageHTML, err := d.FetchPage()
	if err != nil {
		log.Fatalf("Failed to fetch page: %v\n", err)
	}

	links := d.ExtractLinks(pageHTML)
	log.Printf("Found %d total links\n", len(links))

	resources := d.FilterResources(links)
	log.Printf("Found %d resource links\n", len(resources))

	if len(resources) > 0 {
		fmt.Println("\nResources to download:")
		fmt.Println(strings.Repeat("-", 80))
		maxShow := 20
		if len(resources) < maxShow {
			maxShow = len(resources)
		}
		for i := 0; i < maxShow; i++ {
			text := resources[i].Text
			if len(text) > 60 {
				text = text[:60]
			}
			fmt.Printf("%d. %s\n   URL: %s\n", i+1, text, resources[i].URL)
		}
		if len(resources) > 20 {
			fmt.Printf("... and %d more\n", len(resources)-20)
		}
		fmt.Println(strings.Repeat("-", 80))
	}

	if !d.config.Yes {
		fmt.Printf("\nDownload %d resources? (y/n): ", len(resources))
		reader := bufio.NewReader(os.Stdin)
		answer, _ := reader.ReadString('\n')
		answer = strings.TrimSpace(strings.ToLower(answer))
		if answer != "y" && answer != "yes" {
			log.Printf("Download cancelled by user\n")
			return
		}
	}

	log.Printf("Starting download of %d resources...\n", len(resources))
	for i, resource := range resources {
		log.Printf("[%d/%d] %s\n", i+1, len(resources), resource.URL)
		d.DownloadFile(resource.URL)
	}

	fmt.Println("\n" + strings.Repeat("=", 80))
	fmt.Printf("Download complete!\n")
	fmt.Printf("Downloaded: %d files\n", len(d.downloadedFiles))
	fmt.Printf("Failed:     %d files\n", len(d.failedFiles))
	fmt.Printf("Location:   %s\n", d.config.OutputDir)

	if len(d.failedFiles) > 0 {
		fmt.Println("\nFailed downloads:")
		maxShow := 5
		if len(d.failedFiles) < maxShow {
			maxShow = len(d.failedFiles)
		}
		for i := 0; i < maxShow; i++ {
			fmt.Printf("  - %s\n", d.failedFiles[i])
		}
	}
}

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `Web Resource Downloader
=======================
Fetches a web page, extracts links, and downloads matching files.

USAGE:
    %s -url <URL> [options]

OPTIONS:
    -url string
        URL of the page to scrape (required)
    -d, -dir string
        Output directory (default "downloads")
    -ext string
        Comma-separated file extensions to download (default: .pdf,.ppt,.pptx,.docx,.doc,.zip,.mp4,.webm,.csv,.txt,.ipynb)
    -keywords string
        Comma-separated URL keywords to match (e.g. "lecture,session")
    -delay duration
        Delay between downloads (default 1s)
    -retries int
        Number of retry attempts per file (default 3)
    -timeout duration
        Overall timeout for a single download attempt (default 5m)
    -stall duration
        Abort download if no bytes received for this duration (default 30s)
    -y  Skip confirmation prompt

EXAMPLES:
    # Download PDFs and ZIPs from any page
    %s -url https://example.com/course -ext .pdf,.zip -d ./materials

    # Match URLs containing keywords
    %s -url https://example.com/course -keywords lecture,session -d ./materials

    # Non-interactive (no confirmation prompt)
    %s -url https://example.com/course -y

BUILD:
    go build -o downloader downloader.go

DEPENDENCIES:
    Go 1.18+ and golang.org/x/net (fetched automatically)
`, os.Args[0], os.Args[0], os.Args[0], os.Args[0])
	}

	var (
		url        string
		ouput      string
		extns      string
		keywords   string
		exclude    string
		delay      time.Duration
		yes        bool
		retries    int
		reqTimeout time.Duration
		stall      time.Duration
	)

	flag.StringVar(&url, "url", "", "URL of the page to scrape (required)")
	flag.StringVar(&ouput, "d", "downloads", "Output directory")
	flag.StringVar(&ouput, "dir", "downloads", "Output directory")
	flag.StringVar(&extns, "ext", strings.Join(DefaultExtensions, ","), "Comma-separated file extensions")
	flag.StringVar(&keywords, "keywords", "", "Comma-separated URL keywords to match")
	flag.StringVar(&exclude, "exclude", "", "Comma-separated URL regex patterns to exclude (case-insensitive)")
	flag.DurationVar(&delay, "delay", 1*time.Second, "Delay between downloads")
	flag.BoolVar(&yes, "y", false, "Skip confirmation prompt")
	flag.IntVar(&retries, "retries", 3, "Number of retry attempts per file")
	flag.DurationVar(&reqTimeout, "timeout", 5*time.Minute, "Overall timeout per download attempt")
	flag.DurationVar(&stall, "stall", 30*time.Second, "Abort download if no bytes received for this duration")
	flag.Parse()

	if url == "" {
		fmt.Fprintln(os.Stderr, "Error: -url is required")
		flag.Usage()
		os.Exit(1)
	}

	// Parse extensions
	exts := strings.Split(extns, ",")
	for i, e := range exts {
		exts[i] = strings.TrimSpace(e)
	}

	// Parse keywords
	var kws []string
	for _, kw := range strings.Split(keywords, ",") {
		kw = strings.TrimSpace(kw)
		if kw != "" {
			kws = append(kws, kw)
		}
	}

	// Parse exclude regex patterns
	var excludes []*regexp.Regexp
	for _, ex := range strings.Split(exclude, ",") {
		ex = strings.TrimSpace(ex)
		if ex == "" {
			continue
		}
		re, err := regexp.Compile("(?i)" + ex)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: invalid -exclude regex %q: %v\n", ex, err)
			os.Exit(1)
		}
		excludes = append(excludes, re)
	}

	cfg := Config{
		URL:            url,
		OutputDir:      ouput,
		Extensions:     exts,
		Keywords:       kws,
		Exclude:        excludes,
		Delay:          delay,
		Yes:            yes,
		Retries:        retries,
		RequestTimeout: reqTimeout,
		StallTimeout:   stall,
	}

	downloader := NewDownloader(cfg)
	downloader.Run()
}
