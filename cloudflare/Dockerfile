FROM ubuntu

RUN apt-get -y update && apt-get -y install curl
RUN curl -L https://github.com/cloudflare/cloudflared/releases/download/2022.6.3/cloudflared-linux-$(dpkg --print-architecture) -o cloudflared
RUN chmod +x /cloudflared


FROM alpine

COPY --from=0 /cloudflared /
ENTRYPOINT ["/cloudflared", "tunnel", "--no-autoupdate", "run"]
