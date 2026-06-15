import re

from django.conf import settings
from django.http import HttpResponsePermanentRedirect
from django.utils.deprecation import MiddlewareMixin


class SecurityMiddleware(MiddlewareMixin):
    # RemovedInDjango40Warning: when the deprecation ends, replace with:
    #   def __init__(self, get_response):
    def __init__(self, get_response=None):
        """
        Implement your code here
        """
        return None

    def process_request(self, request):
        path = request.path.lstrip("/")
        if (self.redirect and not request.is_secure() and
                not any(pattern.search(path)
                        for pattern in self.redirect_exempt)):
            host = self.redirect_host or request.get_host()
            return HttpResponsePermanentRedirect(
                "https://%s%s" % (host, request.get_full_path())
            )

    def process_response(self, request, response):
        if (self.sts_seconds and request.is_secure() and
                'Strict-Transport-Security' not in response):
            sts_header = "max-age=%s" % self.sts_seconds
            if self.sts_include_subdomains:
                sts_header = sts_header + "; includeSubDomains"
            if self.sts_preload:
                sts_header = sts_header + "; preload"
            response['Strict-Transport-Security'] = sts_header

        if self.content_type_nosniff:
            response.setdefault('X-Content-Type-Options', 'nosniff')

        if self.xss_filter:
            response.setdefault('X-XSS-Protection', '1; mode=block')

        if self.referrer_policy:
            # Support a comma-separated string or iterable of values to allow
            # fallback.
            response.setdefault('Referrer-Policy', ','.join(
                [v.strip() for v in self.referrer_policy.split(',')]
                if isinstance(self.referrer_policy, str) else self.referrer_policy
            ))

        return response
