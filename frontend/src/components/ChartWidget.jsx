import { useEffect, useRef } from 'react';
import { createChart } from 'lightweight-charts';

export default function ChartWidget({ data, atr, atrMultiplier, signal }) {
  const chartContainerRef = useRef();
  const chartRef = useRef();

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      autoSize: true,
      layout: { background: { type: 'solid', color: '#06090f' }, textColor: '#8b949e' },
      grid: { vertLines: { visible: false }, horzLines: { color: '#1c2128' } },
      rightPriceScale: { borderColor: '#30363d' },
      timeScale: { 
          borderColor: '#30363d', 
          timeVisible: true,
          rightOffset: 12,
          barSpacing: 6,
          fixLeftEdge: true,
      },
    });
    
    chartRef.current = chart;

    const candleSeries = chart.addCandlestickSeries({
      upColor: '#3fb950', downColor: '#f85149',
      borderVisible: false, wickUpColor: '#3fb950', wickDownColor: '#f85149',
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 }
    });

    const emaSeries = chart.addLineSeries({
      color: '#58a6ff', lineWidth: 1, title: 'EMA 20',
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 }
    });

    const tpSeries = chart.addLineSeries({
      color: '#3fb950', lineWidth: 1, lineStyle: 2, title: 'TP',
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 }
    });

    const slSeries = chart.addLineSeries({
      color: '#f85149', lineWidth: 1, lineStyle: 2, title: 'SL',
      priceFormat: { type: 'price', precision: 4, minMove: 0.0001 }
    });

    if (data?.candles?.length > 0) {
      candleSeries.setData(data.candles);
      
      if (data.markers && data.markers.length > 0) {
        candleSeries.setMarkers(data.markers);
      }
      
      if (data.ema) {
         emaSeries.setData(data.ema);
      }
      
      if (atr && atrMultiplier && data.candles.length > 0 && signal && signal !== 'NEUTRAL' && signal !== 'HOLD' && signal !== 'UNKNOWN') {
        const lastPrice = data.candles[data.candles.length - 1].close;
        const dist = atr * atrMultiplier;
        
        let tpValue, slValue;
        if (signal === 'BUY') {
            tpValue = lastPrice + dist;
            slValue = lastPrice - dist;
        } else if (signal === 'SELL') {
            tpValue = lastPrice - dist;
            slValue = lastPrice + dist;
        }
        
        if (tpValue && slValue) {
            const tpData = data.candles.slice(-15).map(c => ({ time: c.time, value: tpValue }));
            const slData = data.candles.slice(-15).map(c => ({ time: c.time, value: slValue }));
            tpSeries.setData(tpData);
            slSeries.setData(slData);
        }
      } else {
         // Clear TP/SL if signal is neutral
         tpSeries.setData([]);
         slSeries.setData([]);
      }
    }

    return () => chart.remove();
  }, [data, atr, atrMultiplier, signal]);

  return <div ref={chartContainerRef} className="w-full h-full" />;
}
